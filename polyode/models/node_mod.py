from attr import get_run_validators
import torch
import pytorch_lightning as pl

import torch.nn as nn

# import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from polyode.models.ode_utils import NODE
from torchdiffeq import odeint
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score
from polyode.utils import str2bool

torch.autograd.set_detect_anomaly(True)


class NODEmod(nn.Module):
    def __init__(self, Nc, input_dim, hidden_dim, Delta, corr_time, delta_t, method="euler", extended_ode_mode=False, output_fun="mlp", bridge_ode=False, predict_from_cn=False, auto_encoder=False, **kwargs):
        """
        Nc = dimension of the coefficients vector
        output_dim = dimension of the INPUT (but also the output of the reconstruction)
        hidden_dim = hidden_dimension of the NODE part
        Delta = parameter of the measure
        """
        super().__init__()

        self.uncertainty_fun = nn.Sequential(
            nn.Linear(Nc, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.corr_time = corr_time  # time for the correction
        self.hidden_dim = hidden_dim

        self.delta_t = delta_t
        self.method = method

        self.input_dim = input_dim
        self.extended_ode_mode = extended_ode_mode
        self.bridge_ode = bridge_ode
        self.predict_from_cn = predict_from_cn
        self.Nc = Nc
        self.node = nn.Sequential(
            nn.Linear(Nc * input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, Nc * input_dim))

        self.update_fun = nn.GRUCell(input_size= 2 * input_dim, hidden_size= Nc * input_dim)
        self.output_mod = nn.Sequential(
                nn.Linear(Nc * input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim)) 
        
        self.auto_encoder = auto_encoder
        if auto_encoder:
            self.ae_ode_mod = nn.Sequential(
                nn.Linear(Nc * input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, Nc * input_dim))
            self.ae_out_fun = nn.Sequential(
                nn.Linear(Nc * input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def out_fun(self, h):
        return h[..., :self.input_dim]

    def ode_fun(self, t, cn):
        return self.node(cn)

    def reverse_ode_fun(self, t, cn):
        return -self.node(cn)

    def ae_ode_fun(self, t, cn):
        return self.ae_ode_mod(cn)

    def integrate(self, cn, eval_times, ode_function=None, reverse=False, **kwargs):
        """ Integrate the ODE system

        Args:
            cn (_type_): _description_
            eval_times (_type_): _description_
            ode_function (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if reverse:
            if self.auto_encoder:
                h_out = odeint(self.ae_ode_fun, cn, eval_times,
                           method=self.method, options={"step_size": self.delta_t})
                pred = self.ae_out_fun(h_out)
                return h_out, pred
            else:
                h_out = odeint(self.reverse_ode_fun, cn, eval_times,
                           method=self.method, options={"step_size": self.delta_t})
        else:
            h_out = odeint(self.ode_fun, cn, eval_times,
                           method=self.method, options={"step_size": self.delta_t})

        pred = self.out_fun(h_out)
        return h_out, pred

    def backward_ode(self, end_time, start_time, cn, eval_mode=False, eval_times_ = None, **kwargs):

        if eval_mode:
            if eval_times_ is None:
                eval_times = torch.linspace(
                start_time, end_time, steps=50).to(cn.device)
            else:
                eval_times = eval_times_
        else:
            eval_times = torch.Tensor([start_time, end_time]).to(cn.device)

        h_out, pred = self.integrate(
            cn=cn, eval_times=eval_times, reverse=True, **kwargs)

        eval_times = torch.flip(eval_times, (0,))
        return h_out, pred, eval_times

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

        h_out, pred = self.integrate(cn, eval_times, **kwargs)

        return h_out, pred, eval_times

    def forward(self, times, Y, mask, eval_mode=False, bridge_info=None):
        """
        eval mode returns the ode integrations at multiple times in between observations
        """
        h = torch.zeros(Y.shape[0], self.Nc * self.input_dim, device=Y.device)
        current_time = 0
        preds_list = []
        y_traj = []
        times_traj = []
        if len(mask.shape)==2:
            mask = mask[..., None].repeat((1, 1, self.input_dim))

        for i_t, time in enumerate(times):
            h_out, pred,  eval_times = self.forward_ode(
                time, current_time, h, eval_mode=eval_mode)
            h_cn = h_out[-1]
            y_pred = pred[-1]

            preds_list.append(y_pred)
            y_traj.append(pred[1:].permute(1, 0, 2))
            times_traj.append(eval_times[1:])

            h_updated = self.update_fun(torch.cat((Y[:,i_t],mask[:,i_t]), dim=1),h_cn)

            h = h_updated * mask[:, i_t].any(-1)[...,None].float() + \
                h_cn * (1-mask[:, i_t].any(-1)[..., None].float())

            current_time = time

        y_preds = torch.stack(preds_list, 1)
        y_traj = torch.cat(y_traj, 1)
        times_traj = torch.cat(times_traj, 0)

        if self.auto_encoder:
            reverse_eval_times = torch.abs(times-times[-1])
            reverted_h = odeint(self.ae_ode_fun, h, reverse_eval_times, method=self.method, options={
                "step_size": self.delta_t})
            reverted_h = torch.flip(reverted_h, (0,))
            reverted_pred = self.ae_out_fun(reverted_h.permute(1, 0, 2))
        else:
            reverted_pred = None

        return y_preds, y_traj, times_traj, h, h_cn, reverted_pred


class NODE(pl.LightningModule):
    def __init__(
        self,
        # channels,
        lr=0.001,
        hidden_dim=32,
        output_dim=1,
        step_size=0.05,
        weight_decay=0.,
        Delta=5,
        corr_time=0.5,
        uncertainty_mode=False,
        delta_t=0.05,
        method="dopri5",
        extended_ode_mode=False,
        output_fun="mlp",
        direct_classif=False,
        bridge_ode=False,
        auto_encoder=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.node_model = NODEmod(Nc=hidden_dim, input_dim=output_dim, hidden_dim=hidden_dim, Delta=Delta, corr_time=corr_time, delta_t=delta_t,
                                    method=method, extended_ode_mode=extended_ode_mode, output_fun=output_fun, bridge_ode=bridge_ode, auto_encoder=auto_encoder, **kwargs)
        self.output_mod = nn.Sequential(nn.Linear(
            hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.Delta = Delta
        self.uncertainty_mode = uncertainty_mode

        self.direct_classif = direct_classif
        self.bridge_ode = bridge_ode

        self.auto_encoder = auto_encoder

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
        preds, preds_traj, times_traj, cn_embedding, cn_pre_embedding, reverted_pred = self.node_model(
            times, Y, mask, eval_mode=eval_mode, bridge_info=bridge_info)
        return preds, preds_traj, times_traj, cn_embedding, cn_pre_embedding, reverted_pred

    def get_embedding(self, times, Y, mask, eval_mode=False, bridge_info=None):
        preds, preds_traj, times_traj, cn_embedding, cn_pre_embedding, reverted_pred = self(
            times, Y, mask, eval_mode, bridge_info)
        return cn_embedding

    def compute_loss(self, Y, preds, mask, reverted_preds=None):
        if len(mask.shape) == 3:
            mse = ((preds-Y).pow(2)*(mask)).sum() / mask.sum()
        else:
            mse = ((preds-Y).pow(2)*mask[..., None]).mean(-1).sum() / mask.sum()
        if reverted_preds is not None:
            if len(mask.shape)==3:
                mse += ((reverted_preds-Y).pow(2)*(mask)).sum() / mask.sum()
            else:
                mse += ((reverted_preds-Y).pow(2) *
                    mask[..., None]).mean(-1).sum() / mask.sum()
        return mse

    def process_batch(self, batch,  forecast_mode = False):
        if self.bridge_ode:
            times, Y, mask, label, ids, ts, ys, mask_ids = batch
            return times, Y, mask, label, (ids, ts, ys, mask_ids)
        elif forecast_mode:
            times, Y, mask, label, _, Y_past, Y_future, mask_past, mask_future = batch
            return times, Y, mask, label, None, Y_past, Y_future, mask_past, mask_future
        else:
            times, Y, mask, label, _ = batch
            return times, Y, mask, label, None

    def predict_step(self, batch, batch_idx):
        times, Y, mask, label, bridge_info, Y_past, Y_future, mask_past, mask_future = self.process_batch(
            batch, forecast_mode=True)

        # assert len(torch.unique(T)) == T.shape[1]
        # times = torch.sort(torch.unique(T))[0]
        preds, preds_traj, times_traj, _, cn_pre_embeddings, reverted_preds = self(
            times, Y_past, mask_past, bridge_info=bridge_info)
        
        _, _, _, cn_embedding, _, reverted_preds = self(
            times, Y, mask, bridge_info=bridge_info)

        Tmax = times.max()
        backward_window = 5

        mask_rec = (times-Tmax+backward_window)>0
        rec_span = times[mask_rec]
        
        h_out, recs, recs_times = self.node_model.backward_ode(start_time = 0, end_time = 0, eval_times_ = rec_span,
        cn=cn_embedding, eval_mode=True)
        
        recs = torch.permute(torch.flip(recs, (0,)),(1,0,2))
        return {"Y_future":Y_future, "preds":preds, "mask_future":mask_future, "pred_rec":recs,"Y_rec":Y[:,mask_rec,:],"mask_rec":mask[:,mask_rec,:]} 
    
    def training_step(self, batch, batch_idx):

        times, Y, mask, label, bridge_info = self.process_batch(batch)
        # assert len(torch.unique(T)) == T.shape[1]
        # times = torch.sort(torch.unique(T))[0]
        preds, preds_traj, times_traj, cn_embedding, cn_pre_embedding, reverted_preds = self(
            times, Y, mask, bridge_info=bridge_info)

        preds_class = None
        mse = self.compute_loss(Y, preds, mask, reverted_preds)
        self.log("train_loss", mse, on_epoch=True)
        return {"loss": mse}

    def validation_step(self, batch, batch_idx):
        times, Y, mask, label, bridge_info = self.process_batch(batch)
        # assert len(torch.unique(T)) == T.shape[1]
        # times = torch.sort(torch.unique(T))[0]
        preds, preds_traj, times_traj, cn_embedding, cn_pre_embedding, reverted_preds = self(
            times, Y, mask, bridge_info=bridge_info, eval_mode=True)

        preds_class = None
        mse = self.compute_loss(Y, preds, mask,  reverted_preds)
        self.log("val_mse", mse, on_epoch=True)
        self.log("val_loss", mse, on_epoch=True)
        return {"Y": Y, "preds": preds, "T": times, "preds_traj": preds_traj, "times_traj": times_traj, "mask": mask, "label": label, "pred_class": preds_class, "cn_pre_embedding": cn_pre_embedding}

    def validation_epoch_end(self, outputs) -> None:

        T_sample = outputs[0]["T"]
        Y_sample = outputs[0]["Y"]
        mask = outputs[0]["mask"]
        cn_embedding = outputs[0]["cn_pre_embedding"]
        preds_traj = outputs[0]["preds_traj"]
        times_traj = outputs[0]["times_traj"]
        observed_mask = (mask == 1)
        times = T_sample

        Tmax = T_sample.max().cpu().numpy()
        h_out, recs, recs_times = self.node_model.backward_ode(start_time=Tmax.item(
        )-self.Delta, end_time=Tmax.item(), cn=cn_embedding, eval_mode=True)

        if len(observed_mask.shape)==3:
            observed_mask = observed_mask[...,0]
        else:
            observed_mask = observed_mask
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=T_sample[observed_mask[0]].cpu(
        ), y=Y_sample[0, observed_mask[0], 0].cpu(), mode='markers', name='observations'))
        fig.add_trace(go.Scatter(x=times_traj.cpu(),
                                 y=preds_traj[0, :, 0].cpu(), mode='lines', name='interpolations'))
        fig.add_trace(go.Scatter(
            x=recs_times.cpu().numpy(), y=recs[:, 0, 0].cpu().numpy(), mode='lines', name='polynomial reconstruction'))
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
                            default="implicit_adams", help="integration method")
        parser.add_argument('--output_fun', type=str, default="mlp",
                            help="what type of output function to use in the extended ode case")
        parser.add_argument('--uncertainty_mode', type=str2bool, default=False)
        parser.add_argument('--extended_ode_mode',
                            type=str2bool, default=False)
        parser.add_argument('--direct_classif', type=str2bool, default=False)
        parser.add_argument('--bridge_ode', type=str2bool, default=False)
        parser.add_argument('--predict_from_cn', type=str2bool, default=True,
                            help="if true, the losses are computed on the prediction from the driver ode, not the polynomial reconstruction")
        parser.add_argument('--auto_encoder', type=str2bool, default=False)
        return parser


class NODEClassification(pl.LightningModule):
    def __init__(self, lr,
                 hidden_dim,
                 weight_decay,
                 init_model,
                 pre_compute_ode=False,
                 num_dims=1,
                 regression_mode=False,
                 **kwargs
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.embedding_model = init_model
        self.embedding_model.freeze()
        self.pre_compute_ode = pre_compute_ode

        self.regression_mode = regression_mode
        if regression_mode:
            self.loss_class = torch.nn.MSELoss()
            output_dim = 1
        else:
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
            nn.Linear(hidden_dim * num_dims, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, times, Y, mask, coeffs, eval_mode=False):

        if self.pre_compute_ode:
            embeddings = coeffs
        else:
            embeddings = self.embedding_model.get_embedding(
                times, Y, mask)
        preds = self.classif_model(embeddings)
        return preds

    def predict_step(self, batch, batch_idx):
        times, Y, mask, label, embeddings = batch
        preds = self(times, Y, mask, embeddings)
        if self.regression_mode:
            if len(label.shape)==1:
                label = label[:,None]
            loss = self.loss_class(preds,label)
        else:
            if preds.shape[-1] == 1:
                preds = preds[:, 0]
                loss = self.loss_class(preds.double(), label)
            else:
                loss = self.loss_class(preds.double(), label.long())
        return {"Y": Y, "preds": preds, "T": times, "labels": label}

    def training_step(self, batch, batch_idx):
        times, Y, mask, label, embeddings = batch
        preds = self(times, Y, mask, embeddings)
        if self.regression_mode:
            if len(label.shape)==1:
                label = label[:,None]
            loss = self.loss_class(preds,label)
        else:
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
        if self.regression_mode:
            if len(label.shape)==1:
                label = label[:,None]
            loss = self.loss_class(preds,label)
        else:
            if preds.shape[-1] == 1:
                preds = preds[:, 0]
                loss = self.loss_class(preds.double(), label)
            else:
                loss = self.loss_class(preds.double(), label.long())
        self.log("val_loss", loss, on_epoch=True)
        return {"Y": Y, "preds": preds, "T": times, "labels": label}

    def validation_epoch_end(self, outputs):
        if not self.regression_mode:
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
        parser.add_argument('--auto_encoder', type=str2bool, default=False)
        return parser
