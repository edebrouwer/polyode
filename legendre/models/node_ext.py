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



class CNODExtmod(nn.Module):
    def __init__(self, Nc, input_dim, hidden_dim, Delta, corr_time, delta_t, method="euler", extended_ode_mode=False, output_fun="mlp", bridge_ode=False, predict_from_cn=False, **kwargs):
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

        self.extended_ode_mode = extended_ode_mode
        self.bridge_ode = bridge_ode
        self.predict_from_cn = predict_from_cn
        self.Nc = Nc
        self.node = nn.Sequential(
            nn.Linear(2*Nc + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2*Nc + 1))

    def out_fun(self, h):
        return h[..., 0][..., None]

    def ode_fun(self, t, cn):
        return self.node(cn)
    
    def reverse_ode_fun(self, t, cn):
        return -self.node(cn)

    def integrate(self, cn, eval_times, ode_function=None, reverse = False, **kwargs):
        """ Integrate the ODE system

        Args:
            cn (_type_): _description_
            eval_times (_type_): _description_
            ode_function (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if reverse:
            h_out = odeint(self.reverse_ode_fun, cn, eval_times,
                       method=self.method, options={"step_size": self.delta_t})
        else:
            h_out = odeint(self.ode_fun, cn, eval_times,
                       method=self.method, options={"step_size": self.delta_t})

        pred = self.out_fun(h_out)
        return h_out, pred

    def backward_ode(self,end_time, start_time, cn, eval_mode = False, **kwargs):
        
        if eval_mode:
            eval_times = torch.linspace(
                start_time, end_time, steps=50).to(cn.device)
        else:
            eval_times = torch.Tensor([start_time, end_time]).to(cn.device)
        
        h_out, pred = self.integrate(cn = cn, eval_times = eval_times, reverse = True, **kwargs)

        eval_times = torch.flip(eval_times,(0,))
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
        h = torch.cat(
            (torch.zeros(Y.shape[0], self.Nc, device=Y.device), torch.zeros(Y.shape[0], self.Nc+1, device=Y.device)), -1)
        current_time = 0
        preds_list = []
        y_traj = []
        times_traj = []
        for i_t, time in enumerate(times):
            h_out, pred,  eval_times = self.forward_ode(
                time, current_time, h, eval_mode=eval_mode)
            h_cn = h_out[-1]
            y_pred = pred[-1]

            preds_list.append(y_pred)
            y_traj.append(pred[1:].permute(1, 0, 2))
            times_traj.append(eval_times[1:])

            h_updated = torch.cat(( Y[:, i_t], h_cn[...,:-1]), -1)

            h = h_updated * mask[:, i_t][..., None] + \
                h_cn * (1-mask[:, i_t][..., None])

            current_time = time

        y_preds = torch.stack(preds_list, 1)
        y_traj = torch.cat(y_traj, 1)
        times_traj = torch.cat(times_traj, 0)

        return y_preds, y_traj, times_traj, h, h_cn


class NODExt(pl.LightningModule):
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

    def compute_loss(self, Y, preds, mask):
        mse = ((preds-Y).pow(2)*mask[..., None]).mean(-1).sum() / mask.sum()
        return mse

    def process_batch(self, batch):
        if self.bridge_ode:
            times, Y, mask, label, ids, ts, ys, mask_ids = batch
            return times, Y, mask, label, (ids, ts, ys, mask_ids)
        else:
            times, Y, mask, label = batch
            return times, Y, mask, label, None

    def training_step(self, batch, batch_idx):

        times, Y, mask, label, bridge_info = self.process_batch(batch)
        # assert len(torch.unique(T)) == T.shape[1]
        # times = torch.sort(torch.unique(T))[0]
        preds, preds_traj, times_traj, cn_embedding, cn_pre_embedding = self(
            times, Y, mask, bridge_info=bridge_info)

        preds_class = None
        mse = self.compute_loss(Y, preds, mask)
        self.log("train_loss", mse, on_epoch=True)
        return {"loss": mse}

    def validation_step(self, batch, batch_idx):
        times, Y, mask, label, bridge_info = self.process_batch(batch)
        # assert len(torch.unique(T)) == T.shape[1]
        # times = torch.sort(torch.unique(T))[0]
        preds, preds_traj, times_traj, cn_embedding, cn_pre_embedding = self(
            times, Y, mask, bridge_info=bridge_info, eval_mode=True)

        preds_class = None
        mse = self.compute_loss(Y, preds, mask)
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
        h_out, recs, recs_times = self.node_model.backward_ode(start_time = Tmax.item()-self.Delta, end_time = Tmax.item(), cn = cn_embedding, eval_mode = True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=T_sample[observed_mask[0]].cpu(
        ), y=Y_sample[0, observed_mask[0], 0].cpu(), mode='markers', name='observations'))
        fig.add_trace(go.Scatter(x=times_traj.cpu(),
                                 y=preds_traj[0, :, 0].cpu(), mode='lines', name='interpolations'))
        fig.add_trace(go.Scatter(
            x=recs_times.cpu().numpy(), y=recs[:,0,0].cpu().numpy(), mode='lines', name='polynomial reconstruction'))
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
