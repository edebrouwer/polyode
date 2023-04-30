import torch
import pytorch_lightning as pl

import torch.nn as nn

#import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from polyode.models.ode_utils import NODE
from torchdiffeq import odeint
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score
from polyode.utils import str2bool


def evaluate_spline(t, c, x_eval, spline_type = "Hermite"):
    """
    t are the original break points
    c are the coefficients
    x is the evaluation point
    """
    if x_eval >= t[-1]:
        interval_idx = -1
    else:
        interval_idx = torch.where(
            (t > x_eval)*(x_eval >= torch.cat((torch.zeros(1, device=t.device), t[:-1]))))[0]-1
        try:
            interval_idx = interval_idx.item()
        except:
            import ipdb
            ipdb.set_trace()

    if x_eval > (t[-1] + 0.1):  # x_eval must be less than the last break point
        print("Warning, overflow in the integration time")
        y_ = torch.zeros((c.shape[0], c.shape[-1]), device=c.device)
    else:
        if spline_type == "Hermite":
            x_shifted = (x_eval-t[interval_idx])[None]
            y_ = c[:, -1, interval_idx, :] + c[:, -2, interval_idx, :] * x_shifted + c[:, -
                                                                                3, interval_idx, :] * x_shifted**2 + c[:, -4, interval_idx, :] * x_shifted**3
        elif spline_type == "Constant":
            y_ = c[:,0,interval_idx,:]
        elif spline_type == "Linear":
            x_shifted = (x_eval-t[interval_idx])[None]
            y_ = c[:,0,interval_idx,:] * x_shifted + c[:,1,interval_idx,:]
    return y_



class SplineCNODEClass(pl.LightningModule):
    def __init__(self, lr,
                 hidden_dim,
                 weight_decay,
                 Nc,
                 Delta,
                 num_dims,
                 regression_mode = False,
                 spline_type = "Hermite",
                 **kwargs
                 ):

        super().__init__()
        self.save_hyperparameters()

        self.Nc = Nc
        self.Delta = Delta
        self. num_dims = num_dims

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

        self.regression_mode = regression_mode
        self.spline_type = spline_type
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
            nn.Linear(self.num_dims * Nc, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.pre_compute_ode = self.hparams.get("pre_compute_ode", False)

    def ode_fun(self, t, cn, evaluate_fun):
        fun_preds = evaluate_fun(t)
        cn_chunks = torch.chunk(cn, self.num_dims, 1)
        outs = []
        for dim in range(self.num_dims):
            outs.append(torch.matmul(cn_chunks[dim], self.A.T) +
                        self.B[None, :] * fun_preds[:, dim][:, None])
        return torch.cat(outs, 1)

    def integrate_ode(self, times, Y, mask, coeffs):
        def eval_fun(t): return evaluate_spline(times, coeffs, t, spline_type = self.spline_type)
        c0 = torch.zeros(Y.shape[0], self.Nc * self.num_dims, device=Y.device)
        outputs = odeint(lambda t, cn: self.ode_fun(t, cn, eval_fun), c0, times,
                         method=self.hparams["method"], options={"step_size": self.hparams["delta_t"]})

        # only return the embedding at the last observed observation
        last_observed_indices = (torch.arange(mask.shape[1], device=mask.device)[
                                 None, :].repeat(mask.shape[0], 1)*mask).max(1)[1]
        return outputs[last_observed_indices, torch.arange(Y.shape[0]), :]

    def forward(self, times, Y, mask, coeffs, eval_mode=False):
        if self.pre_compute_ode:
            embeddings = coeffs
        else:
            embeddings = self.integrate_ode(times, Y, mask, coeffs)
        preds = self.classif_model(embeddings)
        return preds

    def training_step(self, batch, batch_idx):
        times, Y, mask, label, coeffs = batch
        preds = self(times, Y, mask, coeffs)
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
        times, Y, mask, label, coeffs = batch
        preds = self(times, Y, mask, coeffs)
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
        return {"Y": Y, "preds": preds, "T": times, "labels": label, "coeffs": coeffs}

    def predict_step(self, batch, batch_idx):
        times, Y, mask, label, coeffs = batch
        preds = self(times, Y, mask, coeffs)
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
        return {"Y": Y, "preds": preds, "T": times, "labels": label, "coeffs": coeffs}

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

            times = outputs[0]["T"]
            coeffs = outputs[0]["coeffs"]
            Y_sample = outputs[0]["Y"]

            if not self.pre_compute_ode:
                x_eval = torch.linspace(
                    times[0], times[-1], 1000, device=times.device)
                spline_eval = torch.cat(
                    [evaluate_spline(times, coeffs, x_eval_, spline_type = self.spline_type) for x_eval_ in x_eval], -1)

                # ----- Plotting the filtered trajectories ----
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_eval.cpu(), y=spline_eval.cpu()[
                            0], mode='lines', name='interpolated spline'))
                fig.add_trace(go.Scatter(
                    x=times.cpu(), y=Y_sample[0].cpu(), mode='markers', name='observations'))

                self.logger.experiment.log({"chart": fig})
            # ---------------------------------------------

    def configure_optimizers(self):
        return torch.optim.Adam(self.classif_model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--Nc', type=float, default=32,
                            help="Dimension of the hidden vector")
        parser.add_argument('--Delta', type=float,
                            default=5, help="Memory span")
        parser.add_argument('--delta_t', type=float,
                            default=0.05, help="integration step size")
        parser.add_argument('--method', type=str,
                            default="implicit_adams", help="integration method")
        return parser
