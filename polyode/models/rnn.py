import torch
import pytorch_lightning as pl
import torch.nn as nn

from sklearn.metrics import roc_auc_score, accuracy_score
from polyode.utils import str2bool


class RNN(pl.LightningModule):
    def __init__(self, lr,
                 Nc,
                 output_dim,
                 ** kwargs
                 ):

        super().__init__()
        self.save_hyperparameters()

        self.input_dim = output_dim
        self.Nc = Nc
        self.rnn_cell = torch.nn.GRUCell(
            input_size=output_dim + 2, hidden_size=Nc * output_dim)
        self.output_cell = torch.nn.Sequential(
            nn.Linear(output_dim*Nc, Nc), nn.ReLU(), nn.Linear(Nc, output_dim))

    def forward(self, times, Y, mask, eval_mode=False):
        """
        eval mode returns the ode integrations at multiple times in between observations
        """
        h = torch.zeros(Y.shape[0], self.input_dim * self.Nc, device=Y.device)

        previous_times = torch.zeros(Y.shape[0], device=Y.device)
        #preds_list = []
        #y_traj = []
        #times_traj = []
        #dt = 0.01
        h_list = [h]
        for i_t, time in enumerate(times):

            t0 = previous_times
            t1 = torch.ones(Y.shape[0], device=Y.device) * time
            if i_t == len(times)-1:
                next_t = t1 + t1 - t0
            else:
                next_t = times[mask[:, i_t+1:].argmax(1) + i_t + 1]
            delta_t = t1-t0  # time since last observation
            delta_next = next_t - t1  # time to next observation
            h_updated = self.rnn_cell(
                torch.cat((Y[:, i_t, :], delta_t[:, None], delta_next[:, None]), -1), h)

            h = h_updated * mask[:, i_t][..., None] + \
                h * (1-mask[:, i_t][..., None])

            h_list.append(h)
            previous_times = time * mask[:, i_t] + \
                previous_times * (1-mask[:, i_t])
        h_tensor = torch.stack(h_list, 1)
        preds = self.output_cell(h_tensor)[:, :-1, :]
        return preds, None, None, h

    def get_embedding(self, times, Y, mask, eval_mode=False):
        preds, _, _, h = self(times, Y, mask, eval_mode)
        return h

    def compute_loss(self, Y, preds, mask):
        mse = ((preds-Y).pow(2)*mask[..., None]).mean(-1).sum() / mask.sum()
        return mse

    def process_batch(self, batch):
        times, Y, mask, label, _ = batch
        return times, Y, mask, label, None

    def training_step(self, batch, batch_idx):

        times, Y, mask, label, bridge_info = self.process_batch(batch)
        preds, preds_traj, times_traj, cn_embedding = self(
            times, Y, mask)

        loss = self.compute_loss(Y, preds, mask)

        self.log("train_loss", loss, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        times, Y, mask, label, bridge_info = self.process_batch(batch)
        preds, preds_traj, times_traj, cn_embedding = self(
            times, Y, mask, eval_mode=True)
        loss = self.compute_loss(Y, preds, mask)

        self.log("val_loss", loss, on_epoch=True)
        return {"Y": Y, "preds": preds, "T": times, "mask": mask, "label": label, "cn_embedding": cn_embedding}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--Nc', type=int, default=32,
                            help="Dimension of the hidden vector")
        parser.add_argument('--delta_t', type=float,
                            default=0.05, help="integration step size")
        return parser


class RNNClassification(pl.LightningModule):
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
