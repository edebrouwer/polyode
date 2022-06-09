from attr import get_run_validators
import torch
import pytorch_lightning  as pl

import torch.nn as nn

#import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from legendre.models.ode_utils import NODE
from torchdiffeq import odeint

def get_value_from_cn(cn):
    Nc = cn.shape[-1]
    prod_cn = cn * torch.Tensor([(2*n+1)**0.5 for n in range(Nc)]).to(cn.device)
    return prod_cn.sum(-1)[...,None]

class CNODEmod(nn.Module):
    def __init__(self,Nc,input_dim ,hidden_dim, Delta):
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

        self.corr_time = 0.1 # time for the correction

    def ode_fun(self,t,cn):
        return torch.matmul(cn,self.A.T) + self.B[None,:]*self.node(cn)

    def ode_approx(self,t,cn, t0,t1, y, y_pred):
        y_est = y_pred + (t-t0)*(y-y_pred)/(t1-t0)
        return torch.matmul(cn,self.A.T) + self.B[None,:]*y_est


    def forward(self,end_time, start_time, cn, eval_mode = False):
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
            h_out = h_out[pre_indices] #only report pre-correction hiddens in eval_mode
            eval_times = eval_times[pre_indices]
        else:
            eval_times = torch.Tensor([start_time,end_time-self.corr_time,end_time]).to(cn.device)
            h_out = odeint(self.ode_fun,cn, eval_times) 
            h_pre = h_out[1]
            h_out = torch.stack([h_out[0],h_out[2]])
            eval_times = eval_times[[0,2]]
        return h_out, eval_times, h_pre

    def update(self,end_time,cn,y,y_pred, eval_mode = False):
        t0 = end_time - self.corr_time
        t1 = end_time
        if eval_mode:
            eval_times = torch.linspace(t0,t1,steps = 10).to(cn.device)
        else:
            eval_times = torch.Tensor([t0,t1]).to(cn.device)
        
        h_out = odeint(lambda t,h : self.ode_approx(t,h, t0 = t0, t1 = t1, y = y, y_pred = y_pred), cn, eval_times)
        return h_out, eval_times



class CNODE(pl.LightningModule):
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
        self.node_model = CNODEmod(Nc = hidden_dim,input_dim = 1, hidden_dim = hidden_dim, Delta = 5)
        #self.update_cell = ObsUpdate(output_dim,hidden_dim)
        self.output_mod = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,output_dim))
    
    def forward(self,times, Y, eval_mode = False):
        """
        eval mode returns the ode integrations at multiple times in between observationsOnly 
        """
        h = torch.zeros( (Y.shape[0],self.hidden_dim), device = Y.device)
        current_time = 0
        preds_vec = []
        times_vec = []
        diff_h = 0
        for i_t, time in enumerate(times):
            h_proj, eval_times, h_pre = self.node_model(time,current_time,h, eval_mode = eval_mode)
            preds = get_value_from_cn(h_proj)
            
            #taking care of the observation transition 
            pred_pre = get_value_from_cn(h_pre)
            h_update, update_eval_times = self.node_model.update(time,h_pre,Y[:,i_t,:].float(),pred_pre)        
            h = h_update[-1]

            if eval_mode:
                preds_update = get_value_from_cn(h_update)
                eval_times = torch.cat([eval_times,update_eval_times])
                preds = torch.cat([preds,preds_update])

            preds_vec.append(preds[1:])
            times_vec.append(eval_times[1:])
            current_time = time
        out_pred = torch.cat(preds_vec).permute(1,0,2) # NxTxD
        out_times = torch.cat(times_vec)
        return out_pred, h, out_times
    
    def training_step(self,batch, batch_idx):
        T, Y = batch
        assert len(torch.unique(T)) == T.shape[1]
        times = torch.sort(torch.unique(T))[0]
        preds, embedding, pred_times = self(times,Y)
        loss = (preds-Y).pow(2).mean()
        self.log("train_loss",loss,on_epoch=True)
        return {"loss":loss}

    def validation_step(self,batch, batch_idx):
        T, Y = batch
        assert len(torch.unique(T)) == T.shape[1]
        times = torch.sort(torch.unique(T))[0]
        preds, embedding, pred_times = self(times,Y)
        loss = (preds-Y).pow(2).mean()
        self.log("val_loss",loss,on_epoch=True)
        return {"Y":Y, "preds":preds, "T":T}

    def validation_epoch_end(self, outputs) -> None:
        T_sample = outputs[0]["T"]
        Y_sample = outputs[0]["Y"]

        times = torch.sort(torch.unique(T_sample))[0]
        preds, embedding, pred_times = self(times,Y_sample, eval_mode = True)

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