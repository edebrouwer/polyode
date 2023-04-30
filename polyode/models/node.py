import torch
import pytorch_lightning  as pl

import torch.nn as nn

#import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from polyode.models.ode_utils import NODE

from sklearn.metrics import roc_auc_score

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
    
    def forward(self,times, Y, mask, eval_mode = False):
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

            h_update = self.update_cell(Y[:,i_t,:].float(),h_proj[-1]) 
            h = h_update * mask[:,i_t,None] + h_proj[-1] * (1-mask[:,i_t,None]) #update only the trajectories with observations       
            
            diff_h += (h-h_proj[-1]).pow(2).mean()

            preds_vec.append(preds[1:])
            times_vec.append(eval_times[1:])
            current_time = time
        out_pred = torch.cat(preds_vec).permute(1,0,2) # NxTxD
        out_times = torch.cat(times_vec)
        return out_pred, h, out_times, diff_h

    def compute_loss(self,Y,preds,mask):
        mse = ((preds-Y).pow(2)*mask[...,None]).mean(-1).sum() / mask.sum()
        #loglik = ((2*torch.log(2*torch.pi*pred_uncertainty.pow(2)) + (preds-Y).pow(2)/pred_uncertainty.pow(2))*(mask[...,None])).mean(-1).sum() / mask.sum()
        #if self.uncertainty_mode:
        #    return loglik, mse
        #else:
        #    return mse, mse
        return mse

    def training_step(self,batch, batch_idx):
        T, Y, mask, label = batch
        times = torch.sort(torch.unique(T))[0]
        preds, embedding, pred_times, diff_h = self(times,Y, mask)
        loss_pred = self.compute_loss(Y,preds,mask)
        loss_h = diff_h
        loss = loss_pred + loss_h
        self.log("train_loss",loss,on_epoch=True)
        return {"loss":loss}

    def validation_step(self,batch, batch_idx):
        T, Y, mask, label = batch
        times = torch.sort(torch.unique(T))[0]
        preds, embedding, pred_times, diff_h = self(times,Y, mask)
        loss_pred = self.compute_loss(Y,preds,mask)
        loss_h = diff_h
        loss = loss_pred + 0.1*loss_h
        self.log("val_loss",loss,on_epoch=True)
        return {"Y":Y, "preds":preds, "T":T, "mask":mask}

    def validation_epoch_end(self, outputs) -> None:
        T_sample = outputs[0]["T"]
        Y_sample = outputs[0]["Y"]
        mask_sample = outputs[0]["mask"]

        times = torch.sort(torch.unique(T_sample))[0]
        preds, embedding, pred_times, diff_h = self(times,Y_sample, mask_sample, eval_mode = True)

        # ----- Plotting the filtered trajectories ---- 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = pred_times.cpu(),y = preds[0,:,0].cpu(),mode = 'lines',name='predictions'))
        fig.add_trace(go.Scatter(x = T_sample.cpu()[mask_sample[0]==1],y = Y_sample[0,:,0].cpu()[mask_sample[0]==1],mode = 'markers',name='observations'))
        
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
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--step_size', type=float, default=0.05)
        return parser


class SequentialODEClassification(pl.LightningModule):
    def __init__(self,lr,
        hidden_dim,
        weight_decay,
        init_model,
        **kwargs
    ):

        super().__init__()
        self.save_hyperparameters()
        self.embedding_model = init_model
        self.embedding_model.freeze()
        self.classif_model = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(), nn.Linear(hidden_dim,1))

    def forward(self,times,Y,mask, eval_mode = False):
        _, embedding, _, _ = self.embedding_model(times,Y, mask)
        preds = self.classif_model(embedding)
        return preds

    def training_step(self,batch, batch_idx):
        T, Y, mask, label = batch
        times = torch.sort(torch.unique(T))[0]
        preds = self(times,Y, mask)[:,0]
        loss = torch.nn.BCEWithLogitsLoss()(preds.double(),label)
        self.log("train_loss",loss,on_epoch=True)
        return {"loss":loss}

    def validation_step(self,batch, batch_idx):
        T, Y, mask, label = batch
        times = torch.sort(torch.unique(T))[0]
        preds = self(times,Y, mask)[:,0]
        loss = torch.nn.BCEWithLogitsLoss()(preds.double(),label)
        self.log("val_loss",loss,on_epoch=True)
        return {"Y":Y, "preds":preds, "T":T, "labels":label}
    
    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        auc = roc_auc_score(labels.cpu().numpy(),preds.cpu().numpy())
        self.log("val_auc",auc,on_epoch=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.classif_model.parameters(),lr = self.hparams.lr, weight_decay = self.hparams.weight_decay)
    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        return parser