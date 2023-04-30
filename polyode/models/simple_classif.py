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
from scipy.signal import cont2discrete
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)





class SimpleClassif(pl.LightningModule):
    def __init__(
        self,
        # channels,
        lr=0.001,
        hidden_dim=32,
        output_dim=1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim

        if self.hparams["data_type"] == "pMNIST":
            self.loss_class = torch.nn.CrossEntropyLoss()
            class_output_dim = 10
        elif self.hparams["data_type"] == "Character":
            self.loss_class = torch.nn.CrossEntropyLoss()
            class_output_dim = 20
        else:
            self.loss_class = torch.nn.BCEWithLogitsLoss()
            class_output_dim = 1

        self.output_dim = output_dim
        self.classif_model = nn.Sequential(nn.Linear(
                output_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, class_output_dim))

    def forward(self, times, Y, mask, eval_mode=False):
        last_idx = mask.shape[1]-1-torch.flip(mask,(1,)).argmax(1)
        x = torch.gather(Y,1,last_idx[:,None,...])[:,0]
        preds = self.classif_model(x)
        return preds, None, None, None

    def get_embedding(self, times, Y, mask, eval_mode=False):
        _, _, _, embedding = self(times, Y, mask, eval_mode=eval_mode)
        return embedding

    def process_batch(self, batch):
        times, Y, mask, label, _ = batch
        return times, Y, mask, label, None

    def training_step(self, batch, batch_idx):

        times, Y, mask, label, bridge_info = self.process_batch(batch)
        preds, preds_traj, times_traj, cn_embedding = self(
            times, Y, mask)

        if preds.shape[-1] == 1:
            preds = preds[:, 0]
            loss = self.loss_class(preds.double(), label)
        else:
            loss = self.loss_class(preds.double(), label.long())
        self.log("train_loss", loss, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        times, Y, mask, label, bridge_info = self.process_batch(batch)
        preds, preds_traj, times_traj, cn_embedding = self(
            times, Y, mask, eval_mode=True)

        if preds.shape[-1] == 1:
            preds = preds[:, 0]
            loss = self.loss_class(preds.double(), label)
        else:
            loss = self.loss_class(preds.double(), label.long())

        preds_class = None
        self.log("val_loss", loss, on_epoch=True)
        return {"Y": Y, "preds": preds, "T": times, "mask": mask, "label": label, "pred_class": preds, "cn_embedding": cn_embedding}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["pred_class"] for x in outputs])
        labels = torch.cat([x["label"] for x in outputs])


        if (self.hparams["data_type"] == "pMNIST") or (self.hparams["data_type"] == "Character"):
            preds = torch.nn.functional.softmax(preds, dim=-1).argmax(-1)
            accuracy = accuracy_score(
                labels.long().cpu().numpy(), preds.cpu().numpy())
            self.log("val_acc", accuracy, on_epoch=True)
        else:
            auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
            self.log("val_auc", auc, on_epoch=True)

        return

    def test_step(self, batch, batch_idx):
        times, Y, mask, label, bridge_info = self.process_batch(batch)
        preds, preds_traj, times_traj, cn_embedding = self(
            times, Y, mask, eval_mode=True)

        if preds.shape[-1] == 1:
            preds = preds[:, 0]
            loss = self.loss_class(preds.double(), label)
        else:
            loss = self.loss_class(preds.double(), label.long())

        preds_class = None
        self.log("test_loss", loss, on_epoch=True)
        return {"Y": Y, "preds": preds, "T": times, "mask": mask, "label": label, "pred_class": preds, "cn_embedding": cn_embedding}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["pred_class"] for x in outputs])
        labels = torch.cat([x["label"] for x in outputs])

        if (self.hparams["data_type"] == "pMNIST") or (self.hparams["data_type"] == "Character"):
            preds = torch.nn.functional.softmax(preds, dim=-1).argmax(-1)
            accuracy = accuracy_score(
                labels.long().cpu().numpy(), preds.cpu().numpy())
            self.log("test_acc", accuracy, on_epoch=True)
        else:
            auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
            self.log("test_auc", auc, on_epoch=True)

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
