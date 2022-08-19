from attr import get_run_validators
import torch
import pytorch_lightning as pl

import torch.nn as nn

# import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from legendre.models.ode_utils import NODE
from torchdiffeq import odeint
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score
from legendre.utils import str2bool
from scipy.signal import cont2discrete
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


def get_value_from_cn(cn):
    Nc = cn.shape[-1]
    prod_cn = cn * \
        torch.Tensor([(2*n+1)**0.5 for n in range(Nc)]).to(cn.device)
    return prod_cn.sum(-1)[..., None]


class MemoryCell(nn.Module):
    def __init__(self,input_size, hidden_size, memory_size, memory_input_size):
        super().__init__()

        self.hidden_activation = torch.nn.Tanh()
        self.hidden_proj = nn.Linear(hidden_size, hidden_size)
        self.update_hidden = nn.Linear(memory_size * memory_input_size + input_size +1, hidden_size)
        self.pre_memory_fun = nn.Linear(input_size +1 + hidden_size, memory_input_size)

        self.output_model = nn.Sequential(nn.Linear(
                hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, input_size))

    def forward(self,input,state,t0,t1,delta_to_next_t,mask, update_memory_fun):
        (h, m) = state
        
        input = torch.cat((input, delta_to_next_t[:,None]), dim=1)
        memory_input =  self.pre_memory_fun(torch.cat((input,h),-1))

        m_updated = update_memory_fun(m,memory_input,t0,t1)

        hidden_pre_updated = self.update_hidden(torch.cat((input,m.reshape(m.shape[0],-1)),-1))
        hidden_updated = self.hidden_activation(hidden_pre_updated + self.hidden_proj(h))

        #import ipdb; ipdb.set_trace()
        hidden = hidden_updated * mask[..., None] + \
                h * (1-mask[..., None])

        memory = m_updated * mask[..., None, None] + \
                m * (1-mask[..., None, None]) 

        outputs = self.output_model(hidden)
        
        return outputs, (hidden, memory)
        



def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures.
    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    if measure == 'tlagt':
        # beta = 1 corresponds to no tilt
        b = measure_args.get('beta', 1.0)
        A = (1.-b)/2 * np.eye(N) - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    if measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) -
                   ss.gammaln(np.arange(N)+1)))
        A = (1./L[:, None]) * A * L[None, :]
        B = (1./L[:, None]) * B * np.exp(-.5 *
                                         ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        Delta = measure_args.get("Delta")
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None] / Delta
        A = -A / Delta
    # LMU: equivalent to LegT up to normalization
    elif measure == 'lmu':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1)[:, None]  # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
        B = (-1.)**Q[:, None] * R
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]

    return A, B


class HIPPOmod(nn.Module):
    def __init__(self, Nc, input_dim, Delta, direct_classif, **kwargs):
        """
        Nc = dimension of the coefficients vector
        output_dim = dimension of the INPUT (but also the output of the reconstruction)
        hidden_dim = hidden_dimension of the NODE part
        Delta = parameter of the measure
        """
        super().__init__()

        A, B = transition(measure="legt", N=Nc, Delta=Delta)
        self.A = nn.Parameter(torch.Tensor(A), requires_grad=False)
        self.B = nn.Parameter(torch.Tensor(B[:, 0]), requires_grad=False)
        self.I = nn.Parameter(torch.eye(Nc), requires_grad=False)

        # C = np.ones((1, Nc))
        # D = np.zeros((1,))
        # A = self.A.detach().numpy()
        # B = self.B.detach().numpy()
        # A, B, _, _, _ = cont2discrete(
        #    (A, B, C, D), dt=0.01, method='bilinear')

        # self.A = nn.Parameter(torch.Tensor(A), requires_grad=False)
        # self.B = nn.Parameter(torch.Tensor(B), requires_grad=False)

        self.Nc = Nc
        self.input_dim = input_dim
        self.direct_classif = direct_classif
        self.hidden_size = Nc
        if not self.direct_classif:
            self.memory_cell = MemoryCell(input_size = input_dim, hidden_size = self.hidden_size, memory_size = self.Nc, memory_input_size = self.input_dim)

    def forward_mult(self, u, delta, precompute=False):
        """ Computes (I + d A) u
        A: (n, n)
        u: (b1* d, n) d represents memory_size
        delta: (b2*, d) or scalar
          Assume len(b2) <= len(b1)
        output: (broadcast(b1, b2)*, d, n)
        """

        if isinstance(delta, torch.Tensor):
            delta = delta.unsqueeze(-1)
        x = F.linear(u, self.A)
        x = u + delta * x

        return x

    def inverse_mult(self, u, delta, precompute=False):
        """ Computes (I - d A)^-1 u """

        if isinstance(delta, torch.Tensor):
            delta = delta.unsqueeze(-1).unsqueeze(-1)

        _A = self.I - delta*self.A
        x = torch.triangular_solve(u.unsqueeze(-1), _A, upper=False)[0]
        x = x[..., 0]

        return x

    def bilinear(self, dt, u, v, alpha=.5, **kwargs):
        """ Computes the bilinear (aka trapezoid or Tustin's) update rule.
        (I - d/2 A)^-1 (I + d/2 A) u + d B (I - d/2 A)^-1 B v
        """
        x = self.forward_mult(u, (1-alpha)*dt, **kwargs)
        v = dt * v
        v = v.unsqueeze(-1) * self.B
        x = x + v
        x = self.inverse_mult(x, (alpha)*dt, **kwargs)
        return x

    def update_memory(self, m, u, t0, t1):
        """
        m: (B, M, N) [batch, memory_size, memory_order]
        u: (B, M)
        t0: (B,) previous time
        t1: (B,) current time
        """

        if torch.eq(t1, 0.).any():
            return F.pad(u.unsqueeze(-1), (0, self.memory_order - 1))
        else:
            dt = ((t1-t0)/t1).unsqueeze(-1)
            m = self.bilinear(dt, m, u)
        return m


    def forward(self,times, Y, mask, eval_mode=False):
        if self.direct_classif:
            return self.forward_hippo(times, Y, mask, eval_mode)
        else:
            return self.forward_hippo_rnn(times, Y, mask, eval_mode)

    def forward_hippo(self, times, Y, mask, eval_mode=False):
        """
        eval mode returns the ode integrations at multiple times in between observations
        """
        h = torch.zeros(Y.shape[0], self.input_dim, self.Nc, device=Y.device)

        previous_times = torch.zeros(Y.shape[0], device=Y.device)
        #preds_list = []
        #y_traj = []
        #times_traj = []
        #dt = 0.01
        for i_t, time in enumerate(times):

            t0 = previous_times
            t1 = torch.ones(Y.shape[0], device=Y.device) * time
            h_updated = self.update_memory(h, Y[:, i_t, :], t0, t1)

            h = h_updated * mask[:, i_t][..., None, None] + \
                h * (1-mask[:, i_t][..., None, None])

            previous_times = time * mask[:, i_t] + \
                previous_times * (1-mask[:, i_t])

        return None, None, None, h

    def forward_hippo_rnn(self,times,Y, mask, eval_mode = False):
        """
        eval mode returns the ode integrations at multiple times in between observations
        """
        m = torch.zeros(Y.shape[0], self.input_dim, self.Nc, device=Y.device)
        h = torch.zeros(Y.shape[0], self.hidden_size, device=Y.device)

        state = (h,m)
        outputs_list = []
        previous_times = torch.zeros(Y.shape[0], device=Y.device)
        for i_t, time in enumerate(times):

            t0 = previous_times
            t1 = torch.ones(Y.shape[0], device=Y.device) * time
            
            if i_t == len(times)-1:
                next_t = t1 + t1 - t0
            else:
                next_t = times[mask[:, i_t+1:].argmax(1) + i_t + 1]
            
            delta_next = next_t - t1  # time to next observation
            outputs, state = self.memory_cell(Y[:,i_t,:], state,  t0, t1, delta_next, mask[:,i_t], update_memory_fun= self.update_memory)
            outputs_list.append(outputs)

            previous_times = time * mask[:, i_t] + \
                previous_times * (1-mask[:, i_t])
        outputs = torch.stack(outputs_list, dim=1)

        (h,m) = state
        return outputs, None, None, m



class HIPPO(pl.LightningModule):
    def __init__(
        self,
        # channels,
        lr=0.001,
        hidden_dim=32,
        output_dim=1,
        step_size=0.05,
        Delta=5,
        direct_classif=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.hippo_model = HIPPOmod(
            Nc=hidden_dim, input_dim=output_dim, hidden_dim=hidden_dim, Delta=Delta, direct_classif= direct_classif, **kwargs)
        # self.output_mod = nn.Sequential(nn.Linear(
        #    hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.Delta = Delta

        self.direct_classif = direct_classif

        if self.hparams["data_type"] == "pMNIST":
            self.loss_class = torch.nn.CrossEntropyLoss()
            class_output_dim = 10
        elif self.hparams["data_type"] == "Character":
            self.loss_class = torch.nn.CrossEntropyLoss()
            class_output_dim = 20
        else:
            self.loss_class = torch.nn.BCEWithLogitsLoss()
            class_output_dim = 1

        self.direct_classif = direct_classif
        self.output_dim = output_dim
        if direct_classif:
            self.classif_model = nn.Sequential(nn.Linear(
                hidden_dim * output_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, class_output_dim))

    def forward(self, times, Y, mask, eval_mode=False):
        preds, _, _, embedding = self.hippo_model(
            times, Y, mask, eval_mode=eval_mode)
        embedding = embedding.reshape(embedding.shape[0], -1)
        if self.direct_classif:
            preds = self.classif_model(embedding)
        else:
            preds = preds
        return preds, None, None, embedding

    def get_embedding(self, times, Y, mask, eval_mode=False):
        _, _, _, embedding = self(times, Y, mask, eval_mode=eval_mode)
        return embedding

    def process_batch(self, batch):
        times, Y, mask, label, _ = batch
        return times, Y, mask, label, None

    def compute_mse_loss(self, Y, preds, mask):
        mse = ((preds[:,:-1]-Y[:,1:]).pow(2)*mask[:,1:, None]).mean(-1).sum() / mask.sum()
        return mse

    def training_step(self, batch, batch_idx):

        times, Y, mask, label, bridge_info = self.process_batch(batch)
        preds, preds_traj, times_traj, cn_embedding = self(
            times, Y, mask)

        if self.direct_classif:
            if preds.shape[-1] == 1:
                preds = preds[:, 0]
                loss = self.loss_class(preds.double(), label)
            else:
                loss = self.loss_class(preds.double(), label.long())
            return {"loss": loss}
        else:
            loss = self.compute_mse_loss(Y,preds,mask)
            return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        times, Y, mask, label, bridge_info = self.process_batch(batch)
        preds, preds_traj, times_traj, cn_embedding = self(
            times, Y, mask, eval_mode=True)

        if self.direct_classif:
            if preds.shape[-1] == 1:
                preds = preds[:, 0]
                loss = self.loss_class(preds.double(), label)
            else:
                loss = self.loss_class(preds.double(), label.long())

        else:
            loss = self.compute_mse_loss(Y,preds,mask)

        preds_class = None
        self.log("val_loss", loss, on_epoch=True)
        return {"Y": Y, "preds": preds, "T": times, "mask": mask, "label": label, "pred_class": preds, "cn_embedding": cn_embedding}

    def validation_epoch_end(self, outputs):
        if self.direct_classif:
            preds = torch.cat([x["pred_class"] for x in outputs])
            labels = torch.cat([x["label"] for x in outputs])
            cn_embedding = outputs[0]["cn_embedding"]
            cn_embedding = torch.stack(torch.chunk(
                cn_embedding, self.output_dim, -1), -1)

            T_sample = outputs[0]["T"]
            Y_sample = outputs[0]["Y"]
            mask = outputs[0]["mask"]

            observed_mask = (mask == 1)
            times = T_sample

            if (self.hparams["data_type"] == "pMNIST") or (self.hparams["data_type"] == "Character"):
                preds = torch.nn.functional.softmax(preds, dim=-1).argmax(-1)
                accuracy = accuracy_score(
                    labels.long().cpu().numpy(), preds.cpu().numpy())
                self.log("val_acc", accuracy, on_epoch=True)
            else:
                auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
                self.log("val_auc", auc, on_epoch=True)

            Tmax = T_sample.max().cpu().numpy()
            Nc = cn_embedding.shape[1]  # number of coefficients
            rec_span = np.linspace(Tmax-self.Delta, Tmax)
            recs = [np.polynomial.legendre.legval(
                (2/self.Delta)*(rec_span-Tmax) + 1, (cn_embedding[..., out_dim].cpu().numpy() * [(2*n+1)**0.5 for n in range(Nc)]).T) for out_dim in range(self.output_dim)]

            dim_to_plot = 0
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=T_sample[observed_mask[0]].cpu(
            ), y=Y_sample[0, observed_mask[0], dim_to_plot].cpu(), mode='markers', name='observations'))
            fig.add_trace(go.Scatter(
                x=rec_span, y=recs[dim_to_plot][0], mode='lines', name='polynomial reconstruction'))
            self.logger.experiment.log({"chart": fig})
        return

    def predict_step(self, batch, batch_idx):
        times, Y, mask, label, bridge_info = self.process_batch(batch)
        preds, _, _, embedding = self(times, Y, mask, eval_mode=False)
        if preds.shape[-1] == 1:
            preds = preds[:, 0]
            loss = self.loss_class(preds.double(), label)
        else:
            loss = self.loss_class(preds.double(), label.long())
        return {"Y": Y, "preds": preds, "T": times, "labels": label}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @ classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--Delta', type=float,
                            default=5, help="Memory span")
        parser.add_argument('--direct_classif',
                            type=str2bool, default=True)
        return parser


class HippoClassification(pl.LightningModule):
    def __init__(self, lr,
                 Nc,
                 init_model,
                 pre_compute_ode=False,
                 num_dims=1,
                 ** kwargs
                 ):

        super().__init__()
        self.save_hyperparameters()

        self.embedding_model = init_model
        self.embedding_model.freeze()
        self.pre_compute_ode = pre_compute_ode

        if self.hparams["data_type"] == "pMNIST":
            self.loss_class = torch.nn.CrossEntropyLoss()
            output_dim = 10
        elif self.hparams["data_type"] == "Character":
            self.loss_class = torch.nn.CrossEntropyLoss()
            output_dim = 20
        else:
            self.loss_class = torch.nn.BCEWithLogitsLoss()
            output_dim = 1

        self.classif_model = nn.Sequential(
            nn.Linear(Nc * num_dims, Nc), nn.ReLU(), nn.Linear(Nc, output_dim))

    def forward(self, times, Y, mask, coeffs, eval_mode=False):

        if self.pre_compute_ode:
            embeddings = coeffs
        else:
            embeddings = self.embedding_model.get_embedding(
                times, Y, mask)
            import ipdb; ipdb.set_trace()
            embedding = embedding.reshape(embedding.shape[0], -1)
        preds = self.classif_model(embeddings)
        return preds

    def predict_step(self, batch, batch_idx):
        times, Y, mask, label, embeddings = batch
        preds = self(times, Y, mask, embeddings)
        if preds.shape[-1] == 1:
            preds = preds[:, 0]
            loss = self.loss_class(preds.double(), label)
        else:
            loss = self.loss_class(preds.double(), label.long())
        return {"Y": Y, "preds": preds, "T": times, "labels": label}

    def training_step(self, batch, batch_idx):
        times, Y, mask, label, embeddings = batch
        preds = self(times, Y, mask, embeddings)
        if preds.shape[-1] == 1:
            preds = preds[:, 0]
            loss = self.loss_class(preds.double(), label)
        else:
            loss = self.loss_class(preds.double(), label.long())
        self.log("train_loss", loss, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        times, Y, mask, label, embeddings = batch
        preds = self(times, Y, mask, embeddings)
        if preds.shape[-1] == 1:
            preds = preds[:, 0]
            loss = self.loss_class(preds.double(), label)
        else:
            loss = self.loss_class(preds.double(), label.long())
        self.log("val_loss", loss, on_epoch=True)
        return {"Y": Y, "preds": preds, "T": times, "labels": label}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        if (self.hparams["data_type"] == "pMNIST") or (self.hparams["data_type"] == "Character"):
            preds = torch.nn.functional.softmax(preds, dim=-1).argmax(-1)
            accuracy = accuracy_score(
                labels.long().cpu().numpy(), preds.cpu().numpy())
            self.log("val_acc", accuracy, on_epoch=True)
        else:
            auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
            self.log("val_auc", auc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.classif_model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--Nc', type=int, default=32,
                            help="Dimension of the hidden vector")
        return parser