from attr import get_run_validators
import torch
import pytorch_lightning  as pl

import torch.nn as nn

#import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from legendre.models.ode_utils import NODE
from torchdiffeq import odeint
import numpy as np

from sklearn.metrics import roc_auc_score
from legendre.utils import str2bool

def get_value_from_cn(cn):
    Nc = cn.shape[-1]
    prod_cn = cn * torch.Tensor([(2*n+1)**0.5 for n in range(Nc)]).to(cn.device)
    return prod_cn.sum(-1)[...,None]

class CNODEmod(nn.Module):
    def __init__(self,Nc,input_dim ,hidden_dim, Delta, corr_time):
        """
        Nc = dimension of the coefficients vector
        output_dim = dimension of the INPUT (but also the output of the reconstruction)
        hidden_dim = hidden_dimension of the NODE part
        Delta = parameter of the measure
        """
        super().__init__()
        
        self.A = nn.Parameter(torch.ones((Nc,Nc)),requires_grad = False)
        self.B = nn.Parameter(torch.ones(Nc,requires_grad = False),requires_grad = False)
        for n in range(Nc):
            self.B[n] = (1/Delta) * ((2*n+1)**0.5)
            for k in range(Nc):
                if k<=n:
                    self.A[n,k] = - (1/Delta)*((2*n+1)**(0.5))*((2*k+1)**(0.5)) * 1
                else:
                    self.A[n,k] = - (1/Delta)*((2*n+1)**(0.5))*((2*k+1)**(0.5)) * (-1)**(n-k)

        self.node = nn.Sequential(nn.Linear(Nc,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,input_dim))

        self.uncertainty_fun = nn.Sequential(nn.Linear(Nc,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,1),nn.Sigmoid())

        self.corr_time = corr_time # time for the correction
        self.hidden_dim = hidden_dim

    def ode_fun(self,t,cn):
        return torch.matmul(cn,self.A.T) + self.B[None,:]*self.node(cn)

    def ode_approx(self,t,cn, t0,t1, y, y_pred):
        y_est = y_pred + (t-t0)*(y-y_pred)/(t1-t0)
        return torch.matmul(cn,self.A.T) + self.B[None,:]*y_est


    def forward_ode(self,end_time, start_time, cn, eval_mode = False):
        if eval_mode:
            """
            When in eval mode, the NODE returns the results in between the observations
            """
            
            eval_times = torch.linspace(start_time,end_time, steps = 10).to(cn.device)
            pre_indices = torch.where(eval_times<end_time-self.corr_time)[0]
            post_indices = torch.where(eval_times>=end_time-self.corr_time)[0]
            eval_times_ = torch.cat([eval_times[pre_indices],torch.Tensor([end_time-self.corr_time]).to(end_time.device),eval_times[post_indices]])

            pre_h_index = len(pre_indices)
            h_out = odeint(self.ode_fun,cn, eval_times_) 
            h_pre = h_out[pre_h_index]
            eval_times_post = eval_times[post_indices]
            h_out_post = h_out[post_indices+1]
            h_out = h_out[pre_indices] #only report pre-correction hiddens in eval_mode
            eval_times = eval_times[pre_indices]
        else:
            eval_times = torch.Tensor([start_time,end_time-self.corr_time,end_time]).to(cn.device)
            h_out = odeint(self.ode_fun,cn, eval_times) 
            h_pre = h_out[1]
            h_out = torch.stack([h_out[0],h_out[2]])
            eval_times = eval_times[[0,2]]
            eval_times_post = None
            h_out_post = None
        return h_out, eval_times, h_pre, eval_times_post, h_out_post

    def update(self,end_time,cn,y,y_pred, eval_mode = False):
        """
        y_pred is the predicted value at end_time - corr_time
        """
        t0 = end_time - self.corr_time 
        t1 = end_time
        if eval_mode:
            eval_times = torch.linspace(t0,t1,steps = 10).to(cn.device)
        else:
            eval_times = torch.Tensor([t0,t1]).to(cn.device)
        
        h_out = odeint(lambda t,h : self.ode_approx(t,h, t0 = t0, t1 = t1, y = y, y_pred = y_pred), cn, eval_times)
        return h_out, eval_times

    
    def forward(self,times, Y, eval_mode = False):
        """
        eval mode returns the ode integrations at multiple times in between observationsOnly 
        """
        h = torch.zeros((Y.shape[0],self.hidden_dim), device = Y.device)
        current_time = 0
        non_corrected_preds = []
        non_corrected_uncertainty = []
        non_corrected_times = []
        preds_vec = []
        uncertainty_vec = []
        times_vec = []
        preds_g_vec = []
        diff_h = 0
        for i_t, time in enumerate(times):
            h_proj, eval_times, h_pre, eval_times_post, h_post = self.forward_ode(time,current_time,h, eval_mode = eval_mode)
            preds = get_value_from_cn(h_proj)
            uncertainty = self.uncertainty_fun(h_proj)
            
            #taking care of the observation transition 
            pred_pre = get_value_from_cn(h_pre)
            #pred_post = get_value_from_cn(h_proj)
            h_update, update_eval_times = self.update(time,h_pre,Y[:,i_t,:].float(),pred_pre)        
            h = h_update[-1]

            if eval_mode:
                preds_post = get_value_from_cn(h_post)
                uncertainty_post = self.uncertainty_fun(h_post)

                non_corrected_preds.append(torch.cat([preds,preds_post])[1:])
                non_corrected_uncertainty.append(torch.cat([uncertainty,uncertainty_post])[1:])
                non_corrected_times.append(torch.cat([eval_times,eval_times_post])[1:])
                
                preds_update = get_value_from_cn(h_update)
                eval_times = torch.cat([eval_times,update_eval_times])
                preds = torch.cat([preds,preds_update])

                preds_g_pre = self.node(h_proj)
                preds_g_post = self.node(h_post)
                preds_g = torch.cat([preds_g_pre,preds_g_post])
                preds_g_vec.append(preds_g[1:])


            preds_vec.append(preds[1:])
            times_vec.append(eval_times[1:])
            uncertainty_vec.append(uncertainty[1:])
            current_time = time
        out_pred = torch.cat(preds_vec).permute(1,0,2) # NxTxD
        out_times = torch.cat(times_vec)
        out_uncertainty = torch.cat(uncertainty_vec).permute(1,0,2)

        if eval_mode:
            non_corrected_times = torch.cat(non_corrected_times)
            non_corrected_preds = torch.cat(non_corrected_preds).permute(1,0,2)
            non_corrected_uncertainty = torch.cat(non_corrected_uncertainty).permute(1,0,2)
            return out_pred, h, out_times, non_corrected_preds, non_corrected_times, preds_g, non_corrected_uncertainty
        return out_pred, h, out_times, out_uncertainty


class CNODE(pl.LightningModule):
    def __init__(
        self,
        #channels,
        lr,
        hidden_dim,
        output_dim,
        step_size,
        weight_decay,
        Delta,
        corr_time,
        uncertainty_mode,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.node_model = CNODEmod(Nc = hidden_dim,input_dim = 1, hidden_dim = hidden_dim, Delta = Delta, corr_time = corr_time)
        #self.update_cell = ObsUpdate(output_dim,hidden_dim)
        self.output_mod = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,output_dim))
        self.Delta = Delta 
        self.uncertainty_mode = uncertainty_mode
    
    def forward(self,times,Y,eval_mode = False):
        return self.node_model(times,Y, eval_mode = eval_mode)

    def compute_loss(self,Y,preds,pred_uncertainty):
        mse = (preds-Y).pow(2).mean()
        loglik = (2*torch.log(2*torch.pi*pred_uncertainty.pow(2)) + (preds-Y).pow(2)/pred_uncertainty.pow(2)).sum(-1).mean()
        if self.uncertainty_mode:
            return loglik
        else:
            return mse

    def training_step(self,batch, batch_idx):
        T, Y, label = batch
        assert len(torch.unique(T)) == T.shape[1]
        times = torch.sort(torch.unique(T))[0]
        preds, embedding, pred_times, pred_uncertainty = self(times,Y)
        loss = self.compute_loss(Y,preds,pred_uncertainty)
        self.log("train_loss",loss,on_epoch=True)
        return {"loss":loss}

    def validation_step(self,batch, batch_idx):
        T, Y, label = batch
        import ipdb; ipdb.set_trace()
        assert len(torch.unique(T)) == T.shape[1]
        times = torch.sort(torch.unique(T))[0]
        preds, embedding, pred_times, pred_uncertainty = self(times,Y)
        loss = self.compute_loss(Y,preds,pred_uncertainty)
        self.log("val_loss",loss,on_epoch=True)
        return {"Y":Y, "preds":preds, "T":T}

    def validation_epoch_end(self, outputs) -> None:
        T_sample = outputs[0]["T"]
        Y_sample = outputs[0]["Y"]

        times = torch.sort(torch.unique(T_sample))[0]
        preds, embedding, pred_times, non_corrected_preds, non_corrected_times, preds_g, non_corrected_uncertainty = self(times,Y_sample, eval_mode = True)

        Tmax = T_sample.max().cpu().numpy()
        Nc = embedding.shape[-1] #number of coefficients
        rec_span = np.linspace(Tmax-self.Delta,Tmax)
        recs = np.polynomial.legendre.legval((2/self.Delta)*(rec_span-Tmax) + 1, (embedding.cpu().numpy() * [(2*n+1)**0.5 for n in range(Nc)]).T)

        # ----- Plotting the filtered trajectories ---- 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = pred_times.cpu(),y = preds[0,:,0].cpu(),mode = 'lines',name='corrected - predictions'))
        fig.add_trace(go.Scatter(x = non_corrected_times.cpu(),y = non_corrected_preds[0,:,0].cpu(),mode = 'lines',name='predictions'))
        fig.add_trace(go.Scatter(x = non_corrected_times.cpu().tolist() + non_corrected_times.cpu().tolist()[::-1],y = (non_corrected_preds[0,:,0].cpu()+non_corrected_uncertainty[0,:,0].cpu()).tolist() + (non_corrected_preds[0,:,0].cpu()-non_corrected_uncertainty[0,:,0].cpu()).tolist()[::-1],fill = "toself",name='predictions uncertainty'))
        fig.add_trace(go.Scatter(x = non_corrected_times.cpu(),y = preds_g[0,:,0].cpu(),mode = 'lines',name='g-predictions'))
        fig.add_trace(go.Scatter(x = rec_span,y = recs[0],mode = 'lines',name='polynomial reconstruction'))
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
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--step_size', type=float, default=0.05)
        parser.add_argument('--Delta', type=float, default=5, help = "Memory span")
        parser.add_argument('--corr_time', type=float, default=0.5, help = "Correction span")
        parser.add_argument('--uncertainty_mode', type=str2bool, default=False)
        return parser


class CNODEClassification(pl.LightningModule):
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

    def forward(self,times,Y,eval_mode = False):
        _, embedding, _ = self.embedding_model(times,Y)
        preds = self.classif_model(embedding)
        return preds

    def training_step(self,batch, batch_idx):
        T, Y, label = batch
        assert len(torch.unique(T)) == T.shape[1]
        times = torch.sort(torch.unique(T))[0]
        preds = self(times,Y)
        loss = torch.nn.BCEWithLogitsLoss()(preds.double(),label)
        self.log("train_loss",loss,on_epoch=True)
        return {"loss":loss}

    def validation_step(self,batch, batch_idx):
        T, Y, label = batch
        assert len(torch.unique(T)) == T.shape[1]
        times = torch.sort(torch.unique(T))[0]
        preds = self(times,Y)
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