import numpy as np
import torch
import pytorch_lightning as pl

from scipy.integrate import odeint
from torch.utils.data import Dataset, DataLoader, Subset
from legendre.utils import str2bool

from legendre.models.spline_cnode import SplineCNODEClass
from scipy.interpolate import CubicHermiteSpline, CubicSpline
import tqdm
from legendre.data_utils.simple_path_utils import get_hermite_spline, collate_irregular_batch


def generate_path(start, Nt, Nobs, irregular_rate=1, regression_mode = False):
    """
    Nt : Number of time points to simulate
    Nobs : Number of observations to select
    phase : phase to add to this particular time series
    irregular_rate : what is the rate of selection of observations (if 1, then the data is regularly sampled)
    """

    def lorenz(x, y, z, s=10, r=28, b=2.667):
        """
        Given:
        x, y, z: a point of interest in three dimensional space
        s, r, b: parameters defining the lorenz attractor
        Returns:
        x_dot, y_dot, z_dot: values of the lorenz attractor's partial
            derivatives at the point x, y, z
        """
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot


    dt = 0.01
    num_steps = 10000

    # Need one more for the initial values
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)

    # Set initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)


    xs = (xs[start:start+Nt]-xs.mean())/xs.std()
    ys = (ys[start:start+Nt]-ys.mean())/ys.std()
    zs = (zs[start:start+Nt]-zs.mean())/zs.std()

    xfull = np.linspace(1, 10, Nt)
    temporal_mask = np.zeros(Nt)
    temporal_mask[::Nt//Nobs] = 1
    xobs = xfull[temporal_mask == 1]

    xsobs = xs[temporal_mask==1]
    ysobs = ys[temporal_mask==1]
    zsobs = zs[temporal_mask==1]

    if regression_mode:
        label = zsobs[int(0.6*Nobs)]
    else:
        label = (zsobs[int(0.6*Nobs)] > 0.).astype(float)
    mask = np.random.binomial(
        1, irregular_rate, size=xobs.shape[0]).astype(bool)
    mask[-1] = 1

    xsobs[~mask] = np.zeros_like(xsobs[~mask])
    ysobs[~mask] = np.zeros_like(ysobs[~mask])
    zsobs[~mask] = np.zeros_like(zsobs[~mask])

    yobs = np.stack([xsobs, ysobs, zsobs], axis=-1)
    return xobs, yobs, label, mask


def generate_dataset(N, Nt, Nobs, irregular_rate, regression_mode = False):
    Xobs = []
    Yobs = []
    labels = []
    masks = []
    for n in range(N):
        start = np.abs(int(np.random.random()*10000-2*Nt))
        xobs, yobs, label, mask = generate_path(
            start, Nt, Nobs, irregular_rate=irregular_rate, regression_mode = regression_mode)

        Xobs.append(xobs)
        Yobs.append(yobs)
        labels.append(label)
        masks.append(mask)
    # lists of the times and observations
    
    return np.stack(Xobs), np.stack(Yobs), np.stack(labels), np.stack(masks)

class LorenzDataset(Dataset):
    def __init__(self, N, Nt=500, Nobs=20, noise_std=0., lorenz_dims = 3,seed=421, irregular_rate=1., spline_mode=False, pre_compute_ode=False, forecast_mode=False,  regression_mode = False, **kwargs):
        super().__init__()
        self.N = N
        self.Nt = Nt
        self.Nobs = Nobs
        self.spline_mode = spline_mode

        self.xobs, self.sequences, self.labels, self.masks = generate_dataset(
                N, Nt, Nobs, irregular_rate=irregular_rate, regression_mode = regression_mode)

        self.sequences = self.sequences[:,:,:lorenz_dims]
        
        if spline_mode:
            self.coeffs = [torch.stack([torch.Tensor(get_hermite_spline(
                self.xobs[n], self.sequences[n, :, dim], self.masks[n, :])) for dim in range(self.sequences.shape[-1])], -1) for n in range(N)]

        self.kwargs = kwargs
        self.pre_compute_ode = pre_compute_ode
        self.num_dims = lorenz_dims
        self.forecast_mode = forecast_mode
        self.regression_mode = regression_mode
        
        self.bridge_ode = False

    def pre_compute_ode_embeddings(self, **kwargs):
        idxs = torch.chunk(torch.arange(self.sequences.shape[0]), 20)
        embedding_list = []
        print("Pre-computing ODE Projection embeddings....")
        if "init_model" in kwargs:
            model = kwargs["init_model"]
            model.eval()
            model.cuda()
            for idx in tqdm.tqdm(idxs):
                embedding = model.get_embedding(torch.Tensor(self.xobs[0]).cuda(), torch.Tensor(
                    self.sequences[idx]).cuda(), torch.Tensor(self.masks[idx]).cuda())
                embedding_list.append(embedding.cpu())
        else:
            if "num_dims" not in self.kwargs:
                self.kwargs["num_dims"] = self.num_dims
            spline_ode = SplineCNODEClass(**self.kwargs).cuda()
            self.coeffs = torch.stack(self.coeffs)
            for idx in tqdm.tqdm(idxs):
                embedding = spline_ode.integrate_ode(torch.Tensor(self.xobs[0]).cuda(), torch.Tensor(
                    self.sequences[idx]).cuda(), torch.Tensor(self.masks[idx]).cuda(), torch.Tensor(self.coeffs[idx]).cuda())
                embedding_list.append(embedding.cpu())
        self.embeddings = torch.cat(embedding_list)
        self.pre_compute_ode = True

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """
        Tobs dim : N x T
        Yobs : N x T x D
        """
        if self.pre_compute_ode:
            return {"Tobs": self.xobs[idx], "Yobs": self.sequences[idx], "mask": self.masks[idx], "label": self.labels[idx], "embeddings": self.embeddings[idx]}
        elif self.spline_mode:
            return {"Tobs": self.xobs[idx], "Yobs": self.sequences[idx], "label": self.labels[idx], "mask": self.masks[idx], "coeffs": self.coeffs[idx]}
        elif self.forecast_mode:
            Y_future = self.sequences[idx].clone()
            mask_future = self.masks[idx].clone()
            mask_past = self.masks[idx].clone()
            Y_past = self.sequences[idx].clone()
            Y_future[(0.8*self.Nobs):] = 0
            Y_past[:(0.8*self.Nobs)] = 0
            mask_future[(0.8*self.Nobs):] = 0
            mask_past[:(0.8*self.Nobs)] = 0
            return {"Tobs": self.xobs[idx], "Yobs": self.sequences[idx], "label": self.labels[idx], "mask": self.masks[idx], "Y_future": Y_future, "mask_future": mask_future, "Y_past": Y_past, "mask_past": mask_past}
        else:
            return {"Tobs": self.xobs[idx], "Yobs": self.sequences[idx], "label": self.labels[idx], "mask": self.masks[idx]}


class LorenzDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, seed=42, N=1000, lorenz_dims = 3, noise_std=0.,  num_workers=4, irregular_rate=1., spline_mode=False, pre_compute_ode=False, forecast_mode=False, Nobs = 20, regression_mode = False,**kwargs):

        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True
        self.noise_std = noise_std
        self.N = N
        self.Nobs = Nobs
        self.irregular_rate = irregular_rate
        self.spline_mode = spline_mode

        self.pre_compute_ode = pre_compute_ode
        self.kwargs = kwargs
        self.num_dims = lorenz_dims
        self.regression_mode = regression_mode

    
    def set_test_only(self):
        self.test_only = True

    def prepare_data(self):

        dataset = LorenzDataset(N=self.N, Nobs = self.Nobs,noise_std=self.noise_std, seed=self.seed, lorenz_dims = self.num_dims,
                                    irregular_rate=self.irregular_rate, spline_mode=self.spline_mode, pre_compute_ode=self.pre_compute_ode, regression_mode = self.regression_mode, **self.kwargs)

        if self.pre_compute_ode:
            dataset.pre_compute_ode_embeddings(**self.kwargs)

        idx = np.random.permutation(len(dataset))
        train_idx = idx[:int(0.5*len(dataset))]
        val_idx = idx[int(0.5*len(dataset)):]
        test_idx = val_idx[int(len(val_idx)/2):]
        val_idx = val_idx[:int(len(val_idx)/2)]

        self.train = Subset(dataset, train_idx)
        self.val = Subset(dataset, val_idx)
        self.test = Subset(dataset, test_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_irregular_batch
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_irregular_batch
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_irregular_batch
        )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--N', type=int, default=1000)
        parser.add_argument('--Nobs', type=int, default=20)
        parser.add_argument('--lorenz_dims', type=int, default=3, help ="number of dimensions of the Lorenz system that are observed" )
        parser.add_argument('--noise_std', type=float, default=0)
        parser.add_argument('--irregular_rate', type=float, default=1.)
        parser.add_argument('--spline_mode', type=str2bool,
                            default=False, help="if True, use spline interpolation")
        parser.add_argument('--pre_compute_ode', type=str2bool, default=False,
                            help="if True, pre-computes the ODE embedding of the splines")
        parser.add_argument('--forecast_mode', type=str2bool, default=False,
                            help="if True, splits the sequence into a past and future part")
        parser.add_argument('--regression_mode', type=str2bool, default=False,
                            help="if True, splits the sequence into a past and future part")
        return parser
