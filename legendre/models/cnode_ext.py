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

torch.autograd.set_detect_anomaly(True)


def get_value_from_cn(cn):
    Nc = cn.shape[-1]
    prod_cn = cn * \
        torch.Tensor([(2*n+1)**0.5 for n in range(Nc)]).to(cn.device)
    return prod_cn.sum(-1)[..., None]


def rk4(func, t0, y0, t_eval, dt):
    vt = torch.zeros(len(t_eval))
    vy = torch.zeros((len(t_eval),) + y0.shape, device=y0.device)
    h = dt
    vt[0] = t = t0
    vy[0] = y = y0
    t_tol = 1e-4
    i_t = 1
    while t < (t_eval[-1] - t_tol):
        h_res = (t_eval[i_t]-t) % dt
        t_next = t_eval[i_t]-h_res
        while t < (t_next - t_tol):
            k1 = h * func(t, y, t_ref=t, y_ref=y)
            k2 = h * func(t + 0.5 * h, y + 0.5 * k1, t_ref=t, y_ref=y)
            k3 = h * func(t + 0.5 * h, y + 0.5 * k2, t_ref=t, y_ref=y)
            k4 = h * func(t + h, y + k3, t_ref=t, y_ref=y)
            t = t + h
            y = y + (k1 + 2*(k2 + k3) + k4) / 6
        assert (t-t_next).abs() < t_tol
        k1 = h * func(t, y, t_ref=t, y_ref=y)
        k2 = h * func(t + 0.5 * h_res, y + 0.5 * k1, t_ref=t, y_ref=y)
        k3 = h * func(t + 0.5 * h_res, y + 0.5 * k2, t_ref=t, y_ref=y)
        k4 = h * func(t + h_res, y + k3, t_ref=t, y_ref=y)
        t = t + h_res
        y = y + (k1 + 2*(k2 + k3) + k4) / 6
        vy[i_t] = y
        vt[i_t] = t
        i_t += 1
        import ipdb
        ipdb.set_trace()
    # EVAL TIMES !
    return vy


class CNODExtmod(nn.Module):
    def __init__(self, Nc, input_dim, hidden_dim, Delta, corr_time, delta_t, method="euler", extended_ode_mode=False, output_fun="mlp", bridge_ode=False, predict_from_cn=False, **kwargs):
        """
        Nc = dimension of the coefficients vector
        output_dim = dimension of the INPUT (but also the output of the reconstruction)
        hidden_dim = hidden_dimension of the NODE part
        Delta = parameter of the measure
        """
        super().__init__()

        self.A = nn.Parameter(torch.ones((Nc, Nc)), requires_grad=False)
        self.B = nn.Parameter(torch.ones(
            Nc, requires_grad=False), requires_grad=False)
        for n in range(Nc):
            self.B[n] = (1/Delta) * ((2*n+1)**0.5)
            for k in range(Nc):
                if k <= n:
                    self.A[n, k] = - (1/Delta)*((2*n+1)**(0.5)
                                                )*((2*k+1)**(0.5)) * 1
                else:
                    self.A[n, k] = - (1/Delta)*((2*n+1)**(0.5)) * \
                        ((2*k+1)**(0.5)) * (-1)**(n-k)

        self.uncertainty_fun = nn.Sequential(
            nn.Linear(Nc, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.corr_time = corr_time  # time for the correction
        self.hidden_dim = hidden_dim

        self.delta_t = delta_t
        self.method = method

        self.extended_ode_mode = extended_ode_mode
        self.bridge_ode = bridge_ode
        self.predict_from_cn = predict_from_cn
        self.Nc = Nc
        self.node = nn.Sequential(
            nn.Linear(Nc + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, Nc + 1))

    def out_fun(self, h):
        return h[..., 0][..., None]

    def ode_fun(self, t, cn):
        cn_0 = cn[:, :self.Nc]
        cn_1 = cn[:, self.Nc:]
        driver_ode = self.node(cn_1)
        cn_ode = torch.matmul(cn_0, self.A.T) + \
            self.B[None, :]*self.out_fun(cn_1)
        return torch.cat((cn_ode, driver_ode), -1)

    def integrate(self, cn, eval_times, ode_function=None, **kwargs):
        """ Integrate the ODE system

        Args:
            cn (_type_): _description_
            eval_times (_type_): _description_
            ode_function (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        h_out = odeint(self.ode_fun, cn, eval_times,
                       method=self.method, options={"step_size": self.delta_t})

        h_cn = h_out[..., :self.Nc]
        h_x = h_out[..., self.Nc:]
        pred = self.out_fun(h_x)
        return h_cn, pred, h_x

    def forward_ode(self, end_time, start_time, cn, eval_mode=False, **kwargs):
        """_summary_

        Args:
            end_time (_type_): _description_
            start_time (_type_): _description_
            cn (_type_): _description_
            eval_mode (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if eval_mode:
            eval_times = torch.linspace(
                start_time, end_time, steps=10).to(cn.device)
        else:
            eval_times = torch.Tensor([start_time, end_time]).to(cn.device)

        h_out, pred, driver_out = self.integrate(cn, eval_times, **kwargs)

        return h_out, pred, driver_out, eval_times

    def forward(self, times, Y, mask, eval_mode=False, bridge_info=None):
        """
        eval mode returns the ode integrations at multiple times in between observations
        """
        h = torch.cat(
            (torch.zeros(Y.shape[0], self.Nc, device=Y.device), torch.zeros(Y.shape[0], self.Nc+1, device=Y.device)), -1)
        current_time = 0
        preds_list = []
        y_traj = []
        times_traj = []
        for i_t, time in enumerate(times):
            h_out, pred, d_out, eval_times = self.forward_ode(
                time, current_time, h, eval_mode=eval_mode)
            h_cn = h_out[-1]
            g_out = d_out[-1]
            y_pred = pred[-1]

            preds_list.append(y_pred)
            y_traj.append(pred[1:].permute(1, 0, 2))
            times_traj.append(eval_times[1:])
            h_no_update = torch.cat((h_cn, g_out), -1)
            h_updated = torch.cat((h_cn, Y[:, i_t], h_cn), -1)

            h = h_updated * mask[:, i_t][..., None] + \
                h_no_update * (1-mask[:, i_t][..., None])

            current_time = time

        y_preds = torch.stack(preds_list, 1)
        y_traj = torch.cat(y_traj, 1)
        times_traj = torch.cat(times_traj, 0)

        return y_preds, y_traj, times_traj, h[..., :self.Nc]


class CNODExt(pl.LightningModule):
    def __init__(
        self,
        # channels,
        lr,
        hidden_dim,
        output_dim,
        step_size,
        weight_decay,
        Delta,
        corr_time,
        uncertainty_mode,
        delta_t,
        method,
        extended_ode_mode,
        output_fun,
        direct_classif=False,
        bridge_ode=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.node_model = CNODExtmod(Nc=hidden_dim, input_dim=1, hidden_dim=hidden_dim, Delta=Delta, corr_time=corr_time, delta_t=delta_t,
                                     method=method, extended_ode_mode=extended_ode_mode, output_fun=output_fun, bridge_ode=bridge_ode, **kwargs)
        self.output_mod = nn.Sequential(nn.Linear(
            hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.Delta = Delta
        self.uncertainty_mode = uncertainty_mode

        self.direct_classif = direct_classif
        self.bridge_ode = bridge_ode

        if self.hparams["data_type"] == "pMNIST":
            self.loss_class = torch.nn.CrossEntropyLoss()
            output_dim = 10
        else:
            self.loss_class = torch.nn.BCEWithLogitsLoss()
            output_dim = 1
        if direct_classif:
            self.classif_model = nn.Sequential(nn.Linear(
                hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, times, Y, mask, eval_mode=False, bridge_info=None):
        return self.node_model(times, Y, mask, eval_mode=eval_mode, bridge_info=bridge_info)

    def get_embedding(self, times, Y, mask, eval_mode=False):
        _, _, _, embedding = self(times, Y, mask, eval_mode=eval_mode)
        return embedding

    def compute_loss(self, Y, preds, mask):
        mse = ((preds-Y).pow(2)*mask[..., None]).mean(-1).sum() / mask.sum()
        return mse

    def process_batch(self, batch):
        if self.bridge_ode:
            times, Y, mask, label, ids, ts, ys, mask_ids = batch
            return times, Y, mask, label, (ids, ts, ys, mask_ids)
        else:
            times, Y, mask, label, _ = batch
            return times, Y, mask, label, None

    def training_step(self, batch, batch_idx):

        times, Y, mask, label, bridge_info = self.process_batch(batch)
        # assert len(torch.unique(T)) == T.shape[1]
        # times = torch.sort(torch.unique(T))[0]
        preds, preds_traj, times_traj, cn_embedding = self(
            times, Y, mask, bridge_info=bridge_info)

        preds_class = None
        mse = self.compute_loss(Y, preds, mask)
        self.log("train_loss", mse, on_epoch=True)
        return {"loss": mse}

    def validation_step(self, batch, batch_idx):
        times, Y, mask, label, bridge_info = self.process_batch(batch)
        # assert len(torch.unique(T)) == T.shape[1]
        # times = torch.sort(torch.unique(T))[0]
        preds, preds_traj, times_traj, cn_embedding = self(
            times, Y, mask, bridge_info=bridge_info, eval_mode=True)

        preds_class = None
        mse = self.compute_loss(Y, preds, mask)
        self.log("val_mse", mse, on_epoch=True)
        self.log("val_loss", mse, on_epoch=True)
        return {"Y": Y, "preds": preds, "T": times, "preds_traj": preds_traj, "times_traj": times_traj, "mask": mask, "label": label, "pred_class": preds_class, "cn_embedding": cn_embedding}

    def validation_epoch_end(self, outputs) -> None:

        T_sample = outputs[0]["T"]
        Y_sample = outputs[0]["Y"]
        mask = outputs[0]["mask"]
        cn_embedding = outputs[0]["cn_embedding"]
        preds_traj = outputs[0]["preds_traj"]
        times_traj = outputs[0]["times_traj"]
        observed_mask = (mask == 1)
        times = T_sample

        Tmax = T_sample.max().cpu().numpy()
        Nc = cn_embedding.shape[-1]  # number of coefficients
        rec_span = np.linspace(Tmax-self.Delta, Tmax)
        recs = np.polynomial.legendre.legval(
            (2/self.Delta)*(rec_span-Tmax) + 1, (cn_embedding.cpu().numpy() * [(2*n+1)**0.5 for n in range(Nc)]).T)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=T_sample[observed_mask[0]].cpu(
        ), y=Y_sample[0, observed_mask[0], 0].cpu(), mode='markers', name='observations'))
        fig.add_trace(go.Scatter(x=times_traj.cpu(),
                                 y=preds_traj[0, :, 0].cpu(), mode='lines', name='interpolations'))
        fig.add_trace(go.Scatter(
            x=rec_span, y=recs[0], mode='lines', name='polynomial reconstruction'))
        self.logger.experiment.log({"chart": fig})
        return

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--step_size', type=float, default=0.05)
        parser.add_argument('--Delta', type=float,
                            default=5, help="Memory span")
        parser.add_argument('--corr_time', type=float,
                            default=0.5, help="Correction span")
        parser.add_argument('--delta_t', type=float,
                            default=0.05, help="integration step size")
        parser.add_argument('--method', type=str,
                            default="dopri5", help="integration method")
        parser.add_argument('--output_fun', type=str, default="mlp",
                            help="what type of output function to use in the extended ode case")
        parser.add_argument('--uncertainty_mode', type=str2bool, default=False)
        parser.add_argument('--extended_ode_mode',
                            type=str2bool, default=False)
        parser.add_argument('--direct_classif', type=str2bool, default=False)
        parser.add_argument('--bridge_ode', type=str2bool, default=False)
        parser.add_argument('--predict_from_cn', type=str2bool, default=True,
                            help="if true, the losses are computed on the prediction from the driver ode, not the polynomial reconstruction")
        return parser


class CNODExtClassification(pl.LightningModule):
    def __init__(self, lr,
                 hidden_dim,
                 weight_decay,
                 init_model,
                 pre_compute_ode=False,
                 **kwargs
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
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, times, Y, mask, coeffs, eval_mode=False):

        if self.pre_compute_ode:
            embeddings = coeffs
        else:
            _, _, _, embedding = self.embedding_model(
                times, Y, mask)
        preds = self.classif_model(embeddings)
        return preds

        preds = self.classif_model(embedding)
        return preds

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
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        return parser
