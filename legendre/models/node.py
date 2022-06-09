import torch
import pytorch_lightning  as pl

import torch.nn as nn

#import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from legendre.models.ode_utils import NODE


class ObsUpdate(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super().__init__()
        self.cell = nn.GRUCell(input_dim,hidden_dim)
    def forward(self,input,hidden):
        return self.cell(input,hidden)

class SequentialODE(pl.LightningModule):
    def __init__(
        self,
        #channels,
        lr,
        hidden_dim,
        output_dim,
        step_size,
        weight_decay,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.node_model = NODE(hidden_dim,n_layers=1)
        self.update_cell = ObsUpdate(output_dim,hidden_dim)
        self.output_mod = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,output_dim))
    
    def forward(self,times, Y, eval_mode = False):
        """
        eval mode returns the ode integrations at multiple times in between observations
        """
        h = torch.zeros( (Y.shape[0],self.hidden_dim), device = Y.device)
        current_time = 0
        preds_vec = []
        times_vec = []
        diff_h = 0
        for i_t, time in enumerate(times):
            h_proj, eval_times = self.node_model(time,current_time,h, eval_mode = eval_mode)
            preds = self.output_mod(h_proj)
            h = self.update_cell(Y[:,i_t,:].float(),h_proj[-1])        
            
            diff_h += (h-h_proj[-1]).pow(2).mean()

            preds_vec.append(preds[1:])
            times_vec.append(eval_times[1:])
            current_time = time
        out_pred = torch.cat(preds_vec).permute(1,0,2) # NxTxD
        out_times = torch.cat(times_vec)
        return out_pred, h, out_times, diff_h
    
    def training_step(self,batch, batch_idx):
        T, Y = batch
        assert len(torch.unique(T)) == T.shape[1]
        times = torch.sort(torch.unique(T))[0]
        preds, embedding, pred_times, diff_h = self(times,Y)
        loss_pred = (preds-Y).pow(2).mean()
        loss_h = diff_h
        loss = loss_pred + loss_h
        self.log("train_loss",loss,on_epoch=True)
        return {"loss":loss}

    def validation_step(self,batch, batch_idx):
        T, Y = batch
        assert len(torch.unique(T)) == T.shape[1]
        times = torch.sort(torch.unique(T))[0]
        preds, embedding, pred_times, diff_h = self(times,Y)
        loss_pred = (preds-Y).pow(2).mean()
        loss_h = diff_h
        loss = loss_pred + 0.1*loss_h
        self.log("val_loss",loss,on_epoch=True)
        return {"Y":Y, "preds":preds, "T":T}

    def validation_epoch_end(self, outputs) -> None:
        T_sample = outputs[0]["T"]
        Y_sample = outputs[0]["Y"]

        times = torch.sort(torch.unique(T_sample))[0]
        preds, embedding, pred_times, diff_h = self(times,Y_sample, eval_mode = True)

        # ----- Plotting the filtered trajectories ---- 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = pred_times.cpu(),y = preds[0,:,0].cpu(),mode = 'lines',name='predictions'))
        fig.add_trace(go.Scatter(x = T_sample[0].cpu(),y = Y_sample[0,:,0].cpu(),mode = 'markers',name='observations'))
        
        self.logger.experiment.log({"chart":fig})
        # ---------------------------------------------

        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = self.hparams.lr, weight_decay = self.hparams.weight_decay)

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.001)
        parser.add_argument('--step_size', type=float, default=0.05)
        return parser