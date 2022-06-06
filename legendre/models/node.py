import torch
import pytorch_lightning  as pl

import torch.nn as nn


class NODE(nn.Module):
    def __init__(self,hidden_dim,output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.node = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim))

    def forward(self,t,x):
        import ipdb; ipdb.set_trace()


class SequentialODE(pl.LightningModule):
    def __init__(
        self,
        #channels,
        lr,
        hidden_dim,
        output_dim,
        weight_decay,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
    
    def forward(self,batch):
        import ipdb; ipdb.set_trace()

    def validation_step(self,data):
        import ipdb; ipdb.set_trace()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = self.hparams.lr, weight_decay = self.hparams.weight_decay)

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.001)
        return parser