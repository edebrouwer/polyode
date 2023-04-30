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


def get_value_from_cn(cn):
    Nc = cn.shape[-1]
    prod_cn = cn * \
        torch.Tensor([(2*n+1)**0.5 for n in range(Nc)]).to(cn.device)
    return prod_cn.sum(-1)[..., None]


def rk4(func, t0, y0, t_eval, dt):
    vt = torch.zeros(len(t_eval))
    vy = torch.zeros((len(t_eval),) + y0.shape, device=y0.device)
    h = dt
    vt[0] = t = t0
    vy[0] = y = y0
    t_tol = 1e-4
    i_t = 1
    while t < (t_eval[-1] - t_tol):
        h_res = (t_eval[i_t]-t) % dt
        t_next = t_eval[i_t]-h_res
        while t < (t_next - t_tol):
            k1 = h * func(t, y, t_ref=t, y_ref=y)
            k2 = h * func(t + 0.5 * h, y + 0.5 * k1, t_ref=t, y_ref=y)
            k3 = h * func(t + 0.5 * h, y + 0.5 * k2, t_ref=t, y_ref=y)
            k4 = h * func(t + h, y + k3, t_ref=t, y_ref=y)
            t = t + h
            y = y + (k1 + 2*(k2 + k3) + k4) / 6
        assert (t-t_next).abs() < t_tol
        k1 = h * func(t, y, t_ref=t, y_ref=y)
        k2 = h * func(t + 0.5 * h_res, y + 0.5 * k1, t_ref=t, y_ref=y)
        k3 = h * func(t + 0.5 * h_res, y + 0.5 * k2, t_ref=t, y_ref=y)
        k4 = h * func(t + h_res, y + k3, t_ref=t, y_ref=y)
        t = t + h_res
        y = y + (k1 + 2*(k2 + k3) + k4) / 6
        vy[i_t] = y
        vt[i_t] = t
        i_t += 1
        import ipdb
        ipdb.set_trace()
    # EVAL TIMES !
    return vy


class CNODExtmod(nn.Module):
    def __init__(self, Nc, input_dim, hidden_dim, Delta, corr_time, delta_t, method="euler", extended_ode_mode=False, output_fun="mlp", bridge_ode=False, predict_from_cn=False, uncertainty_mode=False, **kwargs):
        """
        Nc = dimension of the coefficients vector
        output_dim = dimension of the INPUT (but also the output of the reconstruction)
        hidden_dim = hidden_dimension of the NODE part
        Delta = parameter of the measure
        """
        super().__init__()

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

        self.uncertainty_mode = uncertainty_mode
        if self.uncertainty_mode:
            self.uncertainty_dims = input_dim
            self.uncertainty_mod = nn.Sequential(nn.Linear(Nc*input_dim + input_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,self.uncertainty_dims),nn.Sigmoid())
        else:
            self.uncertainty_dims = 0

        # self.uncertainty_fun = nn.Sequential(
        #    nn.Linear(Nc, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.corr_time = corr_time  # time for the correction
        self.hidden_dim = hidden_dim

        self.delta_t = delta_t
        self.method = method

        self.extended_ode_mode = extended_ode_mode
        self.bridge_ode = bridge_ode
        self.predict_from_cn = predict_from_cn
        self.Nc = Nc
        self.input_dim = input_dim
        self.node = nn.Sequential(
            nn.Linear(input_dim * Nc + input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim * Nc + input_dim ))

    def out_fun(self, h):
        return h[..., :self.input_dim]

    def uncertainty_fun(self, h_in):
        return self.uncertainty_mod(h_in)

    def ode_fun(self, t, cn):
        cn_0 = cn[:, :self.input_dim * self.Nc]
        cn_1 = cn[:, self.input_dim * self.Nc:]
        if self.uncertainty_mode:
            cn_1 = cn[:, self.input_dim * self.Nc:-self.uncertainty_dims * self.Nc]
            cn_uncertainty = cn[:, -self.uncertainty_dims * self.Nc:]
            cn_uncertainty_view = cn_uncertainty.view(
                cn_uncertainty.shape[0], self.input_dim, -1)
            cn_uncertainty = torch.matmul(cn_uncertainty_view, self.A.T) + \
                self.B[None, None, :]*(self.uncertainty_fun(cn_1)[..., None])
            cn_uncertainty = cn_uncertainty.view(cn_uncertainty.shape[0], -1)

        driver_ode = self.node(cn_1)

        cn_0view = cn_0.view(cn_0.shape[0], self.input_dim, -1)
        cn_ode = torch.matmul(cn_0view, self.A.T) + \
            self.B[None, None, :]*(self.out_fun(cn_1)[..., None])
        cn_ode = cn_ode.view(cn_ode.shape[0], -1)
        # cn_ode = torch.matmul(cn_0, self.A.T) + \
        #    self.B[None, :]*self.out_fun(cn_1)
        if self.uncertainty_mode:
            return torch.cat((cn_ode, driver_ode, cn_uncertainty), -1)
        else:
            return torch.cat((cn_ode, driver_ode), -1)

    def integrate(self, cn, eval_times, ode_function=None, **kwargs):
        """ Integrate the ODE system

        Args:
            cn (_type_): _description_
            eval_times (_type_): _description_
            ode_function (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        h_out = odeint(self.ode_fun, cn, eval_times,
                       method=self.method, options={"step_size": self.delta_t})

        h_cn = h_out[..., :self.input_dim * self.Nc]
        h_0 = h_out[...,self.input_dim*self.Nc : self.input_dim*self.Nc + self.input_dim]
        h_x = h_out[..., self.input_dim * self.Nc:]
        pred = self.out_fun(h_x)
        if self.uncertainty_mode:
            h_x = h_out[..., self.input_dim *
                        self.Nc: -self.uncertainty_dims * self.Nc]
            h_cn_uncertainty = h_out[..., -self.uncertainty_dims * self.Nc:]
            pred_uncertainty = self.uncertainty_fun(h_x)
        else:
            h_cn_uncertainty = None
            pred_uncertainty = None
        return h_cn, pred, h_x, pred_uncertainty, h_cn_uncertainty

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

        h_out, pred, driver_out, pred_uncertainty, h_uncertainty = self.integrate(
            cn, eval_times, **kwargs)

        return h_out, pred, driver_out, eval_times, h_uncertainty, pred_uncertainty

    def forward(self, times, Y, mask, eval_mode=False, bridge_info=None):
        """
        eval mode returns the ode integrations at multiple times in between observations
        """
        h = torch.cat(
            (torch.zeros(Y.shape[0], self.input_dim * self.Nc, device=Y.device), torch.zeros(Y.shape[0], self.input_dim * self.Nc + self.input_dim, device=Y.device)), -1)
        last_h = torch.cat(
            (torch.zeros(Y.shape[0], self.input_dim * self.Nc, device=Y.device), torch.zeros(Y.shape[0], self.input_dim * self.Nc + self.input_dim, device=Y.device)), -1)
        if self.uncertainty_mode:
            h = torch.cat((h, torch.zeros(
                Y.shape[0], self.uncertainty_dims * self.Nc, device=Y.device)), -1)
            last_h = torch.cat((last_h, torch.zeros(
                Y.shape[0], self.uncertainty_dims * self.Nc, device=Y.device)), -1)

        current_time = 0
        preds_list = []
        uncertainty_list = []
        uncertainty_traj = []
        y_traj = []
        times_traj = []
        for i_t, time in enumerate(times):
            h_out, pred, d_out, eval_times, h_uncertainty, pred_uncertainty = self.forward_ode(
                time, current_time, h, eval_mode=eval_mode)
            h_cn = h_out[-1]
            g_out = d_out[-1]
            y_pred = pred[-1]

            preds_list.append(y_pred)
            y_traj.append(pred[1:].permute(1, 0, 2))
            times_traj.append(eval_times[1:])

            if self.uncertainty_mode:
                uncertainty_list.append(pred_uncertainty[-1])
                uncertainty_traj.append(pred_uncertainty[1:].permute(1, 0, 2))
                h_uncertainty = h_uncertainty[-1]
                h_no_update = torch.cat((h_cn, g_out, h_uncertainty), -1)
                # exp(-5) is the uncertainty at the observation
                h_updated = torch.cat(
                    (h_cn, Y[:, i_t], h_cn, h_uncertainty), -1)
            else: 
                h_no_update = torch.cat((h_cn, g_out), -1)
                h_updated = torch.cat((h_cn, Y[:, i_t], h_cn), -1)


            if len(mask.shape)==3:
                signal_no_update = g_out[:, :self.input_dim]
                driver_no_update = g_out[:,self.input_dim:].view(g_out.shape[0], self.input_dim,-1).permute(0,2,1)
                driver_update = h_cn.view(h_cn.shape[0], self.input_dim,-1).permute(0,2,1)
                signal = Y[:,i_t] * mask[:,i_t] + signal_no_update * (1-mask[:,i_t])
                driver = (driver_update * mask[:,i_t][:,None,:] + driver_no_update * (1-mask[:,i_t][:,None,:])).reshape(driver_update.shape[0], -1)
                h = torch.cat((h_cn, signal, driver), -1)
                last_h[mask[:,i_t,:].any(1)] = h[mask[:,i_t,:].any(1)]
            else:
                h = h_updated * mask[:, i_t][..., None] + \
                h_no_update * (1-mask[:, i_t][..., None])
                last_h[mask[:, i_t].any()] = h[mask[:, i_t].any()]

            current_time = time

        y_preds = torch.stack(preds_list, 1)
        if self.uncertainty_mode:
            y_uncertainty = torch.stack(uncertainty_list, 1)
            y_uncertainty_traj = torch.cat(uncertainty_traj, 1)
        else:
            y_uncertainty = None
            y_uncertainty_traj = None

        y_traj = torch.cat(y_traj, 1)
        times_traj = torch.cat(times_traj, 0)

        
        if (y_preds.isnan().any()) or (torch.isinf(y_preds).any()):
            import ipdb; ipdb.set_trace()
        return y_preds, y_traj, times_traj, last_h[..., :self.input_dim * self.Nc], h[..., -self.uncertainty_dims*self.Nc:], y_uncertainty, y_uncertainty_traj


class CNODExt(pl.LightningModule):
    def __init__(
        self,
        # channels,
        lr=0.001,
        hidden_dim=32,
        output_dim=1,
        weight_decay=0.,
        Delta=5,
        corr_time=0.5,
        uncertainty_mode=False,
        delta_t=0.05,
        method="dopri5",
        extended_ode_mode=False,
        output_fun="mlp",
        direct_classif=False,
        bridge_ode=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.node_model = CNODExtmod(Nc=hidden_dim, input_dim=output_dim, hidden_dim=hidden_dim, Delta=Delta, corr_time=corr_time, delta_t=delta_t,
                                     method=method, extended_ode_mode=extended_ode_mode, output_fun=output_fun, bridge_ode=bridge_ode, uncertainty_mode=uncertainty_mode, **kwargs)
        # self.output_mod = nn.Sequential(nn.Linear(
        #    hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
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

    def get_embedding(self, times, Y, mask, eval_mode=False):
        _, _, _, embedding, uncertainty_embedding, uncertainty_pred, uncertainty_traj = self(
            times, Y, mask, eval_mode=eval_mode)
        return embedding

    def compute_loss(self, Y, preds, mask, stds=None):
        if stds is not None:
            loss = ((2*torch.log(2*torch.pi*stds.pow(2) + 0.00001) + (preds-Y).pow(2) /
                    (0.001 + stds.pow(2)))*(mask[..., None])).mean(-1).sum() / mask.sum()
        else:
            if len(mask.shape)==3:
                loss = ((preds-Y).pow(2)*(mask)).sum() / mask.sum()
            else:
                loss = ((preds-Y).pow(2)*mask[..., None]
                    ).mean(-1).sum() / mask.sum()
        return loss

    def process_batch(self, batch, forecast_mode=False):
        if self.bridge_ode:
            times, Y, mask, label, ids, ts, ys, mask_ids = batch
            return times, Y, mask, label, (ids, ts, ys, mask_ids)
        elif forecast_mode:
            times, Y, mask, label, _, Y_past, Y_future, mask_past, mask_future = batch
            return times, Y, mask, label, None, Y_past, Y_future, mask_past, mask_future
        else:
            times, Y, mask, label, _ = batch
            return times, Y, mask, label, None

    def predict_step(self, batch, batch_idx):
        times, Y, mask, label, bridge_info, Y_past, Y_future, mask_past, mask_future = self.process_batch(
            batch, forecast_mode=True)

        # assert len(torch.unique(T)) == T.shape[1]
        # times = torch.sort(torch.unique(T))[0]
        preds, preds_traj, times_traj, _, _, _, _ = self(
            times, Y_past, mask_past, bridge_info=bridge_info)
        
        _, _, times_traj, cn_embedding, uncertainty_embedding, uncertainty_pred, uncertainty_traj = self(
            times, Y, mask, bridge_info=bridge_info)
        
        cn_embedding = torch.stack(torch.chunk(
            cn_embedding, self.output_dim, -1), -1)

        
        Tmax = times.max()
        Nc = cn_embedding.shape[1]  # number of coefficients
        backward_window = self.Delta
        mask_rec = (times-Tmax+backward_window)>0
        rec_span = times[mask_rec]
        #rec_span = np.linspace(Tmax-self.Delta, Tmax)
        recs = [np.polynomial.legendre.legval(
            (2/self.Delta)*(rec_span-Tmax).cpu().numpy() + 1, (cn_embedding[..., out_dim].cpu().numpy() * [(2*n+1)**0.5 for n in range(Nc)]).T) for out_dim in range(self.output_dim)]
        recs = torch.Tensor(np.stack(recs,-1))
        
        if self.uncertainty_mode:
            uncertainty_embedding = torch.stack(torch.chunk(
            uncertainty_embedding, self.output_dim, -1), -1)
            uncertainty_recs = [np.polynomial.legendre.legval(
                (2/self.Delta)*(rec_span-Tmax).cpu().numpy() + 1, (uncertainty_embedding[..., out_dim].cpu().numpy() * [(2*n+1)**0.5 for n in range(Nc)]).T) for out_dim in range(self.output_dim)]
            uncertainty_recs = torch.Tensor(np.stack(uncertainty_recs,-1))
        
        return {"Y_future":Y_future, "preds":preds, "mask_future":mask_future, "uncertainty_recs":uncertainty_recs, "uncertainty_pred":uncertainty_pred,"pred_rec":recs,"Y_rec":Y[:,mask_rec,:],"mask_rec":mask[:,mask_rec,...]} 

    def training_step(self, batch, batch_idx):

        times, Y, mask, label, bridge_info = self.process_batch(batch)
        # assert len(torch.unique(T)) == T.shape[1]
        # times = torch.sort(torch.unique(T))[0]
        preds, preds_traj, times_traj, cn_embedding, uncertainty_embedding, uncertainty_pred, uncertainty_traj = self(
            times, Y, mask, bridge_info=bridge_info)

        preds_class = None
        mse = self.compute_loss(Y, preds, mask, stds=uncertainty_pred)
        if (mse.isnan().any()) or torch.isinf(mse):
            import ipdb; ipdb.set_trace()
        self.log("train_loss", mse, on_epoch=True)
        return {"loss": mse}

    def validation_step(self, batch, batch_idx):
        times, Y, mask, label, bridge_info = self.process_batch(batch)
        # assert len(torch.unique(T)) == T.shape[1]
        # times = torch.sort(torch.unique(T))[0]
        preds, preds_traj, times_traj, cn_embedding, uncertainty_embedding, uncertainty_pred, uncertainty_traj = self(
            times, Y, mask, bridge_info=bridge_info, eval_mode=True)

        preds_class = None
        mse = self.compute_loss(Y, preds, mask, stds=uncertainty_pred)
        if (mse.isnan().any()) or torch.isinf(mse):
            import ipdb; ipdb.set_trace()
        mse = mse
        self.log("val_mse", mse, on_epoch=True)
        self.log("val_loss", mse, on_epoch=True)
        return {"Y": Y, "preds": preds, "T": times, "preds_traj": preds_traj, "times_traj": times_traj, "mask": mask, "label": label, "pred_class": preds_class, "cn_embedding": cn_embedding, "uncertainty_pred": uncertainty_pred, "uncertainty_embedding": uncertainty_embedding, "uncertainty_traj": uncertainty_traj}

    
    def validation_epoch_end(self, outputs) -> None:

        T_sample = outputs[0]["T"]
        Y_sample = outputs[0]["Y"]
        mask = outputs[0]["mask"]
        cn_embedding = outputs[0]["cn_embedding"]
        cn_embedding = torch.stack(torch.chunk(
            cn_embedding, self.output_dim, -1), -1)

        preds_traj = outputs[0]["preds_traj"]
        times_traj = outputs[0]["times_traj"]
        observed_mask = (mask == 1)
        if len(mask.shape)==3:
            times = T_sample[mask[0].any(1)]
        else:
            times = T_sample[mask[0].bool()]

        Tmax = times.max().cpu().numpy()
        Nc = cn_embedding.shape[1]  # number of coefficients
        rec_span = np.linspace(Tmax-self.Delta, Tmax)
        recs = [np.polynomial.legendre.legval(
            (2/self.Delta)*(rec_span-Tmax) + 1, (cn_embedding[..., out_dim].cpu().numpy() * [(2*n+1)**0.5 for n in range(Nc)]).T) for out_dim in range(self.output_dim)]

        if self.uncertainty_mode:
            uncertainty_embedding = outputs[0]["uncertainty_embedding"]
            uncertainty_embedding = torch.stack(torch.chunk(
            uncertainty_embedding, self.output_dim, -1), -1)
            uncertainty_pred = outputs[0]["uncertainty_pred"]
            uncertainty_traj = outputs[0]["uncertainty_traj"]
            uncertainty_recs = [np.polynomial.legendre.legval(
                (2/self.Delta)*(rec_span-Tmax) + 1, (uncertainty_embedding[..., out_dim].cpu().numpy() * [(2*n+1)**0.5 for n in range(Nc)]).T) for out_dim in range(self.output_dim)]

        for dim_to_plot in range(Y_sample.shape[-1]):
            
            if len(observed_mask.shape)==3:
                observed_mask_dim = observed_mask[...,dim_to_plot]
            else:
                observed_mask_dim = observed_mask
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=T_sample[observed_mask_dim[0]].cpu(
            ), y=Y_sample[0, observed_mask_dim[0], dim_to_plot].cpu(), mode='markers', name='observations'))
            fig.add_trace(go.Scatter(x=times_traj.cpu(),
                                    y=preds_traj[0, :, dim_to_plot].cpu(), mode='lines', name='interpolations'))
            fig.add_trace(go.Scatter(
                x=rec_span, y=recs[dim_to_plot][0], mode='lines', name='polynomial reconstruction'))
            if self.uncertainty_mode:
                lower_bound = (recs[dim_to_plot][0] -
                            uncertainty_recs[dim_to_plot][0]).tolist()
                upper_bound = (recs[dim_to_plot][0] +
                            uncertainty_recs[dim_to_plot][0]).tolist()
                fig.add_trace(go.Scatter(x=rec_span.tolist() + rec_span.tolist()[
                            ::-1], y=upper_bound + lower_bound[::-1], fill="toself", name='reconstruction uncertainties'))
                fig.add_trace(go.Scatter(x=times_traj.cpu(),
                                        y=uncertainty_traj[0, :, dim_to_plot].cpu(), mode='lines', name='uncertainty preds'))
                fig.add_trace(go.Scatter(x=rec_span,
                                        y=uncertainty_recs[dim_to_plot][0], mode='lines', name='uncertainty recs'))

            self.logger.experiment.log({f"chart_dim_{dim_to_plot}": fig})

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
        parser.add_argument('--Delta', type=float,
                            default=5, help="Memory span")
        parser.add_argument('--corr_time', type=float,
                            default=0.5, help="Correction span")
        parser.add_argument('--delta_t', type=float,
                            default=0.05, help="integration step size")
        parser.add_argument('--method', type=str,
                            default="implicit_adams", help="integration method")
        parser.add_argument('--output_fun', type=str, default="mlp",
                            help="what type of output function to use in the extended ode case")
        parser.add_argument('--uncertainty_mode', type=str2bool, default=False)
        parser.add_argument('--extended_ode_mode',
                            type=str2bool, default=True)
        parser.add_argument('--direct_classif', type=str2bool, default=False)
        parser.add_argument('--bridge_ode', type=str2bool, default=False)
        parser.add_argument('--predict_from_cn', type=str2bool, default=True,
                            help="if true, the losses are computed on the prediction from the driver ode, not the polynomial reconstruction")
        return parser


class MultiLabelCrossEntropyLoss():
    def __init__(self):
        self.loss1 = torch.nn.CrossEntropyLoss()
        self.loss2 = torch.nn.CrossEntropyLoss()

    def __call__(self, inputs, targets):
        import ipdb; ipdb.set_trace()
        return 0.5*(self.loss1(inputs[:,0],targets[:,0].float()) + self.loss2(inputs[:,1],targets[:,1].float()))

class CNODExtClassification(pl.LightningModule):
    def __init__(self, lr,
                 hidden_dim,
                 weight_decay,
                 init_model,
                 pre_compute_ode=False,
                 num_dims=1,
                 regression_mode = False,
                 **kwargs
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.embedding_model = init_model
        self.embedding_model.freeze()
        self.pre_compute_ode = pre_compute_ode

        self.regression_mode = regression_mode
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
            elif self.hparams["data_type"] == "Activity":
                self.loss_class = MultiLabelCrossEntropyLoss()
                output_dim = 6*2
            else:
                self.loss_class = torch.nn.BCEWithLogitsLoss()
                output_dim = 1

        self.classif_model = nn.Sequential(
            nn.Linear(hidden_dim * num_dims, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, times, Y, mask, coeffs, eval_mode=False):

        if self.pre_compute_ode:
            embeddings = coeffs
        else:
            _, _, _, embeddings = self.embedding_model(
                times, Y, mask)
        preds = self.classif_model(embeddings)
        return preds


    def predict_step(self, batch, batch_idx):
        times, Y, mask, label, embeddings = batch
        preds = self(times, Y, mask, embeddings)
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
        return {"Y": Y, "preds": preds, "T": times, "labels": label}

    def training_step(self, batch, batch_idx):
        times, Y, mask, label, embeddings = batch
        preds = self(times, Y, mask, embeddings)
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
        times, Y, mask, label, embeddings = batch
        preds = self(times, Y, mask, embeddings)
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
        return {"Y": Y, "preds": preds, "T": times, "labels": label}

    def validation_epoch_end(self, outputs):
        if not self.regression_mode:
            preds = torch.cat([x["preds"] for x in outputs])
            labels = torch.cat([x["labels"] for x in outputs])
            if (self.hparams["data_type"] == "pMNIST") or (self.hparams["data_type"] == "Character"):
                preds = torch.nn.functional.softmax(preds, dim=-1).argmax(-1)
                accuracy = accuracy_score(
                    labels.long().cpu().numpy(), preds.cpu().numpy())
                self.log("val_acc", accuracy, on_epoch=True)
            elif self.hparams["data_type"] == "Activity" :
                import ipdb; ipdb.set_trace()
                auc1 = roc_auc_score(labels[:,0].cpu().numpy(), preds[:,0].cpu().numpy())
                auc2 = roc_auc_score(labels[:,0].cpu().numpy(), preds[:,1].cpu().numpy())
            else:
                auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
                self.log("val_auc", auc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.classif_model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        return parser
