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

from sklearn.metrics import roc_auc_score, accuracy_score
from legendre.utils import str2bool

torch.autograd.set_detect_anomaly(True)

def get_value_from_cn(cn):
    Nc = cn.shape[-1]
    prod_cn = cn * torch.Tensor([(2*n+1)**0.5 for n in range(Nc)]).to(cn.device)
    return prod_cn.sum(-1)[...,None]


def rk4(func, t0, y0, t_eval, dt):
    vt = torch.zeros(len(t_eval))
    vy = torch.zeros( (len(t_eval),) + y0.shape, device = y0.device)
    h = dt
    vt[0] = t = t0
    vy[0] = y = y0
    t_tol = 1e-4
    i_t = 1
    while t<(t_eval[-1] - t_tol):
        h_res = (t_eval[i_t]-t)%dt
        t_next = t_eval[i_t]-h_res
        while t < (t_next - t_tol):
            k1 = h * func(t, y, t_ref = t, y_ref = y)
            k2 = h * func(t + 0.5 * h, y + 0.5 * k1, t_ref = t, y_ref = y)
            k3 = h * func(t + 0.5 * h, y + 0.5 * k2, t_ref = t, y_ref = y)
            k4 = h * func(t + h, y + k3, t_ref = t, y_ref = y)
            t = t + h
            y = y + (k1 + 2*(k2 + k3) + k4) / 6
        assert (t-t_next).abs()<t_tol
        k1 = h * func(t, y, t_ref = t, y_ref = y)
        k2 = h * func(t + 0.5 * h_res, y + 0.5 * k1, t_ref = t, y_ref = y)
        k3 = h * func(t + 0.5 * h_res, y + 0.5 * k2, t_ref = t, y_ref = y)
        k4 = h * func(t + h_res, y + k3, t_ref = t, y_ref = y)
        t = t + h_res
        y = y + (k1 + 2*(k2 + k3) + k4) / 6
        vy[i_t] = y
        vt[i_t] = t
        i_t += 1
        import ipdb; ipdb.set_trace()
    #EVAL TIMES !
    return vy




class CNODEmod(nn.Module):
    def __init__(self,Nc,input_dim ,hidden_dim, Delta, corr_time, delta_t, method = "euler", extended_ode_mode = False, output_fun = "mlp", bridge_ode = False, predict_from_cn = False, **kwargs):
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


        self.uncertainty_fun = nn.Sequential(nn.Linear(Nc,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,1),nn.Sigmoid())

        self.corr_time = corr_time # time for the correction
        self.hidden_dim = hidden_dim

        self.delta_t = delta_t
        self.method = method

        self.extended_ode_mode = extended_ode_mode
        self.output_fun = output_fun
        self.bridge_ode = bridge_ode
        self.predict_from_cn = predict_from_cn

        if self.extended_ode_mode:
            self.node = nn.Sequential(nn.Linear(Nc,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,Nc))
            if self.output_fun=="from_cn":
                self.out_fun = get_value_from_cn
            else:
                self.out_fun = nn.Sequential(nn.Linear(Nc,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,input_dim))
        elif self.bridge_ode:
            self.bridge_fun = nn.Sequential(nn.Linear(Nc+2 + 2*input_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,input_dim))
        else:
            self.node = nn.Sequential(nn.Linear(Nc,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,input_dim))

    def ode_fun(self,t,cn):
        return torch.matmul(cn,self.A.T) + self.B[None,:]*self.node(cn)

    def ode_fun_ex(self,t,cn,t_ref,y_ref):
        g = get_value_from_cn(cn) + 0.1*torch.tanh(self.node(cn))*(t-t_ref)
        return torch.matmul(cn,self.A.T) + self.B[None,:]*g

    def ode_fun_wide(self,t,cn):
        driver_ode = self.node(cn[...,1])
        cn_ode = torch.matmul(cn[...,0],self.A.T) + self.B[None,:]*self.out_fun(cn[...,1])
        return torch.stack((cn_ode,driver_ode),-1)

    def ode_approx(self,t,cn, t0,t1, y, y_pred):
        y_est = y_pred + (t-t0)*(y-y_pred)/(t1-t0)
        return torch.matmul(cn,self.A.T) + self.B[None,:]*y_est

    def linear(self,t,x0,y0,x1,y1):
        return (t*(y1-y0) + x1*y0 - x0*y1) / (x1-x0)
        #return -((t-x1)/(x1-x0)) * y0 + ((t-x0)/(x1-x0))*y1

    def distance(self,t,t0,t1):
        return torch.min(torch.abs(t0-t),torch.abs(t1-t))

    def bridge_function(self,t,cn,t0,t1,y0,y1):
        cat_input = torch.cat((cn,y0,y1,(t1-t0),(t-t0)),-1)
        #return self.distance(t,t0,t1) + self.linear(t,t0,y0,t1,y1)
        return (self.bridge_fun(cat_input) * self.distance(t,t0,t1)) + self.linear(t,t0,y0,t1,y1)
        #return (self.bridge_fun(cat_input) * self.distance(t,t0,t1) + self.linear(t,t0,y0,t1,y1))

    def ode_bridge_wide(self,t,cn,t0,t1,y0,y1):
        g = self.bridge_function(t,cn,t0,t1,y0,y1)
        #g = torch.sin(t).repeat(cn.shape[0])[:,None]
        return torch.matmul(cn,self.A.T) + self.B[None,:]*g 


    def integrate(self,cn,eval_times, ode_function = None, **kwargs):
        if ode_function is not None:
            h_out = odeint(ode_function, cn, eval_times, method = self.method, options = {"step_size":self.delta_t})
            return h_out,  None
        else:
            if self.extended_ode_mode:
                cn_ext = torch.stack((cn,cn),-1)
                h_out = odeint(self.ode_fun_wide,cn_ext,eval_times,method = self.method, options = {"step_size":self.delta_t})
                return h_out[...,0], self.out_fun(h_out[...,1])
            elif self.bridge_ode:
                h_out = odeint(lambda t,h : self.ode_bridge_wide(t,h,t0 = kwargs["t0"],t1 = kwargs["t1"],y0 = kwargs["y0"],y1 = kwargs["y1"]),cn,eval_times,method = self.method, options = {"step_size":self.delta_t})
                return h_out, None
            else:
                h_out =  odeint(self.ode_fun,cn,eval_times, method = self.method, options = {"step_size":self.delta_t})
                return h_out, self.node(h_out)
           


    def forward_ode(self,end_time, start_time, cn, eval_mode = False, **kwargs):
        if eval_mode:
            """
            When in eval mode, the NODE returns the results in between the observations
            """
            
            eval_times = torch.linspace(start_time,end_time, steps = 10).to(cn.device)
            pre_indices = torch.where(eval_times<end_time-self.corr_time)[0]
            post_indices = torch.where(eval_times>=end_time-self.corr_time)[0]
            eval_times_ = torch.cat([eval_times[pre_indices],torch.Tensor([end_time-self.corr_time]).to(end_time.device),eval_times[post_indices]])

            pre_h_index = len(pre_indices)
            #h_out = odeint(self.ode_fun,cn, eval_times_, method = self.method, options = {"step_size":self.delta_t}) 
            h_out, g_out = self.integrate(cn = cn,eval_times = eval_times_, **kwargs)
            h_pre = h_out[pre_h_index]
            g_pre = g_out[pre_indices]
            eval_times_post = eval_times[post_indices]
            h_out_post = h_out[post_indices+1]
            g_post = g_out[post_indices+1]
            h_out = h_out[pre_indices] #only report pre-correction hiddens in eval_mode
            eval_times = eval_times[pre_indices]
            #g_out = torch.cat((g_pre,g_post))
            g_out = g_out
        else:
            eval_times = torch.Tensor([start_time,end_time-self.corr_time,end_time]).to(cn.device)
            #h_out = odeint(self.ode_fun,cn, eval_times, method =self.method, options = {"step_size":self.delta_t}) 
            h_out, g_out = self.integrate(cn = cn ,eval_times = eval_times, **kwargs)
            h_pre = h_out[1]
            h_out = torch.stack([h_out[0],h_out[2]])
            eval_times = eval_times[[0,2]]
            eval_times_post = None
            h_out_post = None
        return h_out, eval_times, h_pre, eval_times_post, h_out_post, g_out

    def update(self,end_time,cn,y,y_pred, eval_mode = False, eval_times = None):
        """
        y_pred is the predicted value at end_time - corr_time
        In eval_model, eval_times force evaluation times in the update
        """
        t0 = end_time - self.corr_time 
        t1 = end_time
        if eval_mode:
            assert eval_times is not None #if eval_times is not None, we need to provide the eval_times

            eval_times = torch.cat([t0[None],eval_times])
            assert eval_times[0] == t0
            assert eval_times[-1] == t1
            eval_times_ = eval_times
            #eval_times_ = torch.linspace(t0,t1,steps = 10).to(cn.device)
        else:
            eval_times_ = torch.Tensor([t0,t1]).to(cn.device)
        
        #h_out = odeint(lambda t,h : self.ode_approx(t,h, t0 = t0, t1 = t1, y = y, y_pred = y_pred), cn, eval_times_, method = self.method, options={"step_size":self.delta_t})
        h_out, g_out = self.integrate(cn,eval_times_, ode_function = lambda t,h : self.ode_approx(t,h, t0 = t0, t1 = t1, y = y, y_pred = y_pred))
        return h_out, eval_times_

    def compute_bridge_loss(self,h_vec, bridge_info):
        (ids,ts,ys,mask_ids) = bridge_info
        cn_select = h_vec.gather(1,ids[...,0][...,None].repeat(1,1,h_vec.shape[-1]).long())
        t0 = ts[...,0][...,None]
        t1 = ts[...,1][...,None]
        tmid = ts[...,2][...,None]
        y0 = ys[...,0][...,None]
        y1 = ys[...,1][...,None]
        ymid = ys[...,2][...,None]
        cat_input = torch.cat((cn_select,y0,y1,(t1-t0),(tmid-t0)),-1)
        pred_bridge = (self.bridge_fun(cat_input) * self.distance(tmid,t0,t1) + self.linear(tmid,t0,y0,t1,y1)) 
        loss_bridge = (pred_bridge[mask_ids==1]-ymid[mask_ids==1]).pow(2).mean()

        return loss_bridge

    def compute_preds(self,h_proj,g_out,eval_mode = False, post = False, update_idx = 0):
        if self.predict_from_cn:
            preds = get_value_from_cn(h_proj)
        else:
            if eval_mode:
                if post:
                    preds = g_out[:len(h_proj)]
                else:
                    preds = g_out[update_idx]        
            else:
                if post:
                    preds = g_out[[0,2]]
                else:
                    preds = g_out[1]
        return preds
        
    def forward(self,times, Y, mask, eval_mode = False, bridge_info = None):
        """
        eval mode returns the ode integrations at multiple times in between observations 
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
        h_vec = []
        diff_h = 0
        y0 = torch.zeros((Y.shape[0],Y.shape[-1]),device = Y.device)
        t0 = torch.ones((Y.shape[0],1),device = Y.device)*current_time

        for i_t, time in enumerate(times):
            if self.bridge_ode:
                next_obs_time_idx = torch.cat([torch.where(mask[i,i_t:]==1)[0].min()[None] for i in range(mask.shape[0])]) + i_t
                y1 = Y[torch.arange(Y.shape[0]),next_obs_time_idx,:]
                t1 = times[next_obs_time_idx][:,None]
                h_proj, eval_times, h_pre, eval_times_post, h_post, g_out = self.forward_ode(time,current_time,h, eval_mode = eval_mode, y0 =y0, y1 = y1, t0 = t0, t1= t1)
                h_vec.append(h_proj[-1])
                h = h_proj[-1]
            else:
                h_proj, eval_times, h_pre, eval_times_post, h_post, g_out = self.forward_ode(time,current_time,h, eval_mode = eval_mode)
                """
                h_post is the hidden process between (t-corr_time) and t without correction
                h_proj is the hidden process at time t (without the correction)
                h_pre is the hidden right before the correction
                g_out are all predicted inputs at eval_times.
                """
                preds = self.compute_preds(h_proj,g_out,eval_mode = eval_mode,post = True)
                #preds = get_value_from_cn(h_proj)
                uncertainty = self.uncertainty_fun(h_proj)
                
                #taking care of the observation transition 
                pred_pre = self.compute_preds(h_pre,g_out,eval_mode = eval_mode,post = False, update_idx = len(h_proj))
                #pred_pre = get_value_from_cn(h_pre)
                #pred_post = get_value_from_cn(h_proj)

                h_update, update_eval_times = self.update(time,h_pre,Y[:,i_t,:].float(),pred_pre, eval_mode = eval_mode, eval_times = eval_times_post)
                h = h_update[-1] * mask[:,i_t,None] + h_proj[-1] * (1-mask[:,i_t,None]) #update only the trajectories with observations
                #h[mask[:,i_t]==1] = h_update[-1][mask[:,i_t]==1] # update only the trajectories with an observation

                if eval_mode:
                    #preds_post = get_value_from_cn(h_post)
                    preds_post = self.compute_preds(h_post,g_out[len(h_proj)+1:],eval_mode = eval_mode,post = True)
                    uncertainty_post = self.uncertainty_fun(h_post)

                    non_corrected_preds_ = torch.cat([preds,preds_post])[1:]
                    non_corrected_preds.append(non_corrected_preds_)
                    non_corrected_uncertainty.append(torch.cat([uncertainty,uncertainty_post])[1:])
                    non_corrected_times.append(torch.cat([eval_times,eval_times_post])[1:])
                    
                    preds_update = get_value_from_cn(h_update[1:]) #discarding the correction time
                    eval_times = torch.cat([eval_times,update_eval_times[1:]]) #discarding the correction time
                    
                    assert((eval_times_post==update_eval_times[1:]).all())

                    preds_corrected = preds_post.clone()
                    preds_corrected[:,mask[:,i_t]==1] = preds_update[:,mask[:,i_t]==1]
                    preds = torch.cat([preds,preds_corrected])

                    #preds_g_pre = self.node(h_proj)
                    #preds_g_post = self.node(h_post)

                    preds_g = torch.cat([g_out[:len(h_proj)],g_out[len(h_proj)+1:]])
                    preds_g = preds_g[1:]
                    preds_g_vec.append(preds_g)

                
                preds_vec.append(preds[1:])
                times_vec.append(eval_times[1:])
                uncertainty_vec.append(uncertainty[1:])
            
            current_time = time
            if self.bridge_ode:
                update_t0_mask = (current_time == t1)
                t0[update_t0_mask] = t1[update_t0_mask]
                y0[update_t0_mask] = y1[update_t0_mask]
        

        if self.bridge_ode:
            #loss_bridge = h_vec[0].sum()

            h_vec = torch.stack(h_vec).permute(1,0,2)
            loss_bridge = self.compute_bridge_loss(h_vec, bridge_info)
            return h_vec, None, None, None, loss_bridge
        else:
            out_pred = torch.cat(preds_vec).permute(1,0,2) # NxTxD
            out_times = torch.cat(times_vec)
            out_uncertainty = torch.cat(uncertainty_vec).permute(1,0,2)


            if eval_mode:
                non_corrected_times = torch.cat(non_corrected_times)
                non_corrected_preds = torch.cat(non_corrected_preds).permute(1,0,2)
                non_corrected_uncertainty = torch.cat(non_corrected_uncertainty).permute(1,0,2)
                preds_g = torch.cat(preds_g_vec).permute(1,0,2)
                return out_pred, h, out_times, non_corrected_preds, non_corrected_times, preds_g, non_corrected_uncertainty
            return out_pred, h, out_times, out_uncertainty, None


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
        delta_t,
        method,
        extended_ode_mode,
        output_fun,
        direct_classif = False,
        bridge_ode = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.node_model = CNODEmod(Nc = hidden_dim,input_dim = 1, hidden_dim = hidden_dim, Delta = Delta, corr_time = corr_time, delta_t = delta_t, method = method, extended_ode_mode = extended_ode_mode, output_fun = output_fun, bridge_ode = bridge_ode, **kwargs)
        self.output_mod = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,output_dim))
        self.Delta = Delta 
        self.uncertainty_mode = uncertainty_mode
        
        self.direct_classif = direct_classif
        self.bridge_ode = bridge_ode

        if self.hparams["data_type"]=="pMNIST":
            self.loss_class = torch.nn.CrossEntropyLoss()
            output_dim = 10
        else:
            self.loss_class = torch.nn.BCEWithLogitsLoss()
            output_dim = 1
        if direct_classif:
            self.classif_model = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(), nn.Linear(hidden_dim,output_dim))
    
    def forward(self,times,Y, mask, eval_mode = False, bridge_info = None):
        return self.node_model(times,Y, mask, eval_mode = eval_mode, bridge_info = bridge_info)

    def compute_loss(self,Y,preds,pred_uncertainty,mask):
        mse = ((preds-Y).pow(2)*mask[...,None]).mean(-1).sum() / mask.sum()
        loglik = ((2*torch.log(2*torch.pi*pred_uncertainty.pow(2)) + (preds-Y).pow(2)/pred_uncertainty.pow(2))*(mask[...,None])).mean(-1).sum() / mask.sum()
        if self.uncertainty_mode:
            return loglik, mse
        else:
            return mse, mse

    def process_batch(self,batch):
        if self.bridge_ode:
            times, Y, mask, label, ids, ts, ys, mask_ids = batch
            return times, Y, mask, label, (ids, ts, ys, mask_ids)
        else:
            times, Y, mask, label = batch
            return times, Y, mask, label, None
        
    def training_step(self,batch, batch_idx):
        times, Y, mask, label, bridge_info = self.process_batch(batch)
        preds, embedding, pred_times, pred_uncertainty, loss_bridge= self(times,Y, mask, bridge_info = bridge_info)
        
        if self.bridge_ode:
            loss = loss_bridge
            preds_class = None
        else: 
            if self.direct_classif:
                preds_class = self.classif_model(embedding)
                preds_class = torch.nn.functional.softmax(preds_class,dim=-1)
                loss = self.loss_class(preds_class.double(),label.long())
            else:
                loss, mse = self.compute_loss(Y,preds,pred_uncertainty,mask)
        
        self.log("train_loss",loss,on_epoch=True)
        return {"loss":loss}

    def validation_step(self,batch, batch_idx):
        times, Y, mask, label, bridge_info = self.process_batch(batch)
        #assert len(torch.unique(T)) == T.shape[1]
        #times = torch.sort(torch.unique(T))[0]
        preds, embedding, pred_times, pred_uncertainty, loss_bridge = self(times,Y, mask, bridge_info =  bridge_info)
        
        if self.bridge_ode:
            loss = loss_bridge
            preds_class = None
        else:
            if self.direct_classif:
                preds_class = self.classif_model(embedding)
                preds_class = torch.nn.functional.softmax(preds_class,dim=-1)
                loss = self.loss_class(preds_class.double(),label.long())
                accuracy = accuracy_score(label.cpu().numpy(),preds_class.argmax(-1).cpu().numpy())
                self.log("val_acc",accuracy,on_epoch=True)
            else:
                preds_class = None
                loss,mse = self.compute_loss(Y,preds,pred_uncertainty,mask)
                self.log("val_mse",mse,on_epoch = True)
        self.log("val_loss",loss,on_epoch=True)
        return {"Y":Y, "preds":preds, "T":times, "mask":mask, "label":label, "pred_class":preds_class, "bridge_info":bridge_info}

    def validation_epoch_end(self, outputs) -> None:
        
        T_sample = outputs[0]["T"]
        Y_sample = outputs[0]["Y"]
        mask = outputs[0]["mask"]

        observed_mask = (mask==1)
        times = T_sample
        
        if self.bridge_ode:
            bridge_info = outputs[0]["bridge_info"]
            h_vec,_,_,_,_ = self(times,Y_sample,mask,bridge_info = bridge_info)
            id0 = torch.where(mask[0]==1)[0]
            id1 = id0[1:]
            id0 = id0[:-1]
            y0 = Y_sample[0][id0]
            y1 = Y_sample[0][id1]
            t0 = times[id0]
            t1 = times[id1]
            times_eval = []
            y_eval = []
            for id_ in range(len(t1)):
                for t in torch.linspace(t0[id_],t1[id_],steps=10):
                    cat_input = torch.cat((h_vec[0,id_],y0[id_],y1[id_],(t1[id_]-t0[id_])[None],(t-t0[id_])[None]),-1)
                    pred_bridge = (self.node_model.bridge_fun(cat_input) * self.node_model.distance(t,t0[id_],t1[id_]) + self.node_model.linear(t,t0[id_],y0[id_],t1[id_],y1[id_])) 
                    times_eval.append(t)
                    y_eval.append(pred_bridge)
            y_eval = torch.stack(y_eval)[:,0]
            times_eval = torch.stack(times_eval)

            Tmax = T_sample.max().cpu().numpy()
            embedding = h_vec[:,-1]
            Nc = embedding.shape[-1] #number of coefficients
            rec_span = np.linspace(Tmax-self.Delta,Tmax)
            recs = np.polynomial.legendre.legval((2/self.Delta)*(rec_span-Tmax) + 1, (embedding.cpu().numpy() * [(2*n+1)**0.5 for n in range(Nc)]).T)
        
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = T_sample[observed_mask[0]].cpu(),y = Y_sample[0,observed_mask[0],0].cpu(),mode = 'markers',name='observations'))
            fig.add_trace(go.Scatter(x = times_eval.cpu(),y = y_eval.cpu(),mode = 'lines',name='interpolations'))
            fig.add_trace(go.Scatter(x = rec_span,y = recs[0],mode = 'lines',name='polynomial reconstruction'))
            self.logger.experiment.log({"chart":fig})
            return

        else:
            if self.direct_classif:
                preds = torch.cat([x["pred_class"] for x in outputs])
                labels = torch.cat([x["label"] for x in outputs])
                accuracy = accuracy_score(labels.cpu().numpy(),preds.argmax(-1).cpu().numpy())
                self.log("val_acc",accuracy,on_epoch=True)

            preds, embedding, pred_times, non_corrected_preds, non_corrected_times, preds_g, non_corrected_uncertainty = self(times,Y_sample,mask, eval_mode = True)

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
            fig.add_trace(go.Scatter(x = T_sample[observed_mask[0]].cpu(),y = Y_sample[0,observed_mask[0],0].cpu(),mode = 'markers',name='observations'))
            
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
        parser.add_argument('--delta_t', type=float, default=0.05, help = "integration step size")
        parser.add_argument('--method', type=str, default="dopri5", help = "integration method")
        parser.add_argument('--output_fun', type=str, default="mlp", help = "what type of output function to use in the extended ode case")
        parser.add_argument('--uncertainty_mode', type=str2bool, default=False)
        parser.add_argument('--extended_ode_mode', type=str2bool, default=False)
        parser.add_argument('--direct_classif', type=str2bool, default=False)
        parser.add_argument('--bridge_ode', type=str2bool, default=False)
        parser.add_argument('--predict_from_cn', type=str2bool, default=True,help="if true, the losses are computed on the prediction from the driver ode, not the polynomial reconstruction")
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

    def forward(self,times,Y,mask,eval_mode = False):
        _, embedding, _, pred_uncertainty = self.embedding_model(times,Y,mask)
        preds = self.classif_model(embedding)
        return preds

    def training_step(self,batch, batch_idx):
        times, Y, mask, label = batch
        preds = self(times,Y,mask)[:,0]
        loss = torch.nn.BCEWithLogitsLoss()(preds.double(),label)
        self.log("train_loss",loss,on_epoch=True)
        return {"loss":loss}

    def validation_step(self,batch, batch_idx):
        times, Y, mask, label = batch
        preds = self(times,Y,mask)[:,0]
        loss = torch.nn.BCEWithLogitsLoss()(preds.double(),label)
        self.log("val_loss",loss,on_epoch=True)
        return {"Y":Y, "preds":preds, "T":times, "labels":label}
    
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