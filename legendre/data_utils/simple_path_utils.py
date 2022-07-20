import numpy as np
import torch
import pytorch_lightning as pl

from scipy.integrate import odeint
from torch.utils.data import Dataset, DataLoader, Subset
from legendre.utils import str2bool

from scipy.interpolate import CubicHermiteSpline, CubicSpline

def generate_path(phase,Nt,Nobs, irregular_rate = 1):
    """
    Nt : Number of time points to simulate
    Nobs : Number of observations to select
    phase : phase to add to this particular time series
    irregular_rate : what is the rate of selection of observations (if 1, then the data is regularly sampled)
    """
    x = np.linspace(1,10,Nt)
    y = np.sin(x + phase) * np.cos(3*(x+phase))
    xobs = x[1:-1:Nt//Nobs]
    yobs = y[1:-1:Nt//Nobs]

    label = (yobs[5]>0.5).astype(float)
    mask = np.random.binomial(1,irregular_rate,size = xobs.shape[0]).astype(bool)
    yobs[~mask] = np.zeros_like(yobs[~mask])

    return x,y, xobs, yobs, label, mask

def generate_dataset(N,Nt,Nobs, irregular_rate ):
    Xobs = []
    Yobs = []
    labels = []
    masks = []
    for n in range(N):
        phase = 2*np.random.randn()*np.pi
        x,y,xobs, yobs,label, mask = generate_path(phase,Nt,Nobs, irregular_rate = irregular_rate)
         
        Xobs.append(xobs)
        Yobs.append(yobs[...,None])
        labels.append(label)
        masks.append(mask)
    return np.stack(Xobs), np.stack(Yobs), np.stack(labels), np.stack(masks) # lists of the times and observations


def get_hermite_spline(xobs,yobs,mask):

    dydx = np.concatenate([(yobs[mask][1:]-yobs[mask][:-1])/(xobs[mask][1:]-xobs[mask][:-1]),np.zeros(1)])
    spline = CubicHermiteSpline(x=xobs[mask],y=yobs[mask],dydx= dydx)

    y_ = spline(xobs[~mask])
    diff_ = spline(xobs[~mask],nu=1)
    ynew = np.zeros(mask.shape)
    dydx_new = np.zeros(mask.shape)
    ynew[mask] = yobs[mask]
    ynew[~mask] = y_
    dydx_new[mask] = dydx
    dydx_new[~mask] = diff_
    spline_new = CubicHermiteSpline(x=xobs,y=ynew,dydx= dydx_new)
    return spline_new.c


def generate_spline_dataset(N,Nt,Nobs,irregular_rate):
    Xobs = []
    Yobs = []
    labels = []
    masks = []
    Coeffs = []
    for n in range(N):
        phase = 2*np.random.randn()*np.pi
        x,y,xobs, yobs,label, mask = generate_path(phase,Nt,Nobs, irregular_rate = irregular_rate)
        
        coeffs = get_hermite_spline(xobs,yobs,mask)

        Xobs.append(xobs)
        Yobs.append(yobs)
        Coeffs.append(coeffs)
        labels.append(label)
        masks.append(mask)
    return np.stack(Xobs), np.stack(Yobs), np.stack(labels), np.stack(masks), np.stack(Coeffs) # lists of the times and observations 

def collate_irregular_batch(batch):
    T_obs = [b["Tobs"] for b in batch]
    Y_obs = np.stack([b["Yobs"] for b in batch])
    masks = np.stack([b["mask"] for b in batch])
    labels = torch.Tensor([b["label"] for b in batch])

    unique_times = np.unique(np.concatenate(T_obs))
    used_times = masks.sum(0)>0
    assert((masks.sum(0)>0).all())
    unique_times = unique_times[used_times] #check if some times are never used.

    Y_obs = Y_obs[:,used_times]
    masks = masks[:,used_times]

    if "coeffs" in batch[0]: 
        coeffs = np.stack([b["coeffs"] for b in batch])
        if "embeddings" in batch[0]:
            embeddings = np.stack([b["embeddings"] for b in batch])
            return torch.Tensor(unique_times), torch.Tensor(Y_obs), torch.Tensor(masks), labels, torch.Tensor(embeddings)
        else:
            return torch.Tensor(unique_times), torch.Tensor(Y_obs), torch.Tensor(masks), labels, torch.Tensor(coeffs)
    else:
        return torch.Tensor(unique_times), torch.Tensor(Y_obs), torch.Tensor(masks), labels


class SimpleTrajDataset(Dataset):
    def __init__(self,N,Nt = 200,Nobs = 10, noise_std = 0., seed = 421, irregular_rate = 1., spline_mode = False):
        super().__init__()
        self.N = N
        self.Nt = Nt
        self.Nobs = Nobs
        self.spline_mode = spline_mode

        np.random.seed(seed)
        if spline_mode:
            self.Tobs, self.Yobs, self.labels, self.masks, self.coeffs = generate_spline_dataset(N,Nt,Nobs, irregular_rate = irregular_rate)
        else:
            self.Tobs, self.Yobs, self.labels, self.masks = generate_dataset(N,Nt,Nobs, irregular_rate = irregular_rate)

    def __len__(self):
        return self.N
    
    def __getitem__(self,idx):
        """
        Tobs dim : N x T
        Yobs : N x T x D
        """
        if self.spline_mode:
            return {"Tobs":self.Tobs[idx], "Yobs":self.Yobs[idx], "label":self.labels[idx], "mask":self.masks[idx], "coeffs":self.coeffs[idx]}
        else:
            return {"Tobs":self.Tobs[idx], "Yobs":self.Yobs[idx], "label":self.labels[idx], "mask":self.masks[idx]}


class SimpleTrajDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, seed, N,  noise_std,  num_workers = 4, irregular_rate = 1., spline_mode = False,  **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True
        self.noise_std = noise_std
        self.N = N
        self.irregular_rate = irregular_rate
        self.spline_mode = spline_mode

    def prepare_data(self):

        dataset = SimpleTrajDataset(N = self.N, noise_std =  self.noise_std, seed = self.seed, irregular_rate= self.irregular_rate, spline_mode = self.spline_mode)       
        
        train_idx = np.arange(len(dataset))[:int(0.5*len(dataset))]
        val_idx = np.arange(len(dataset))[int(0.5*len(dataset)):]
        test_idx = val_idx[int(len(val_idx)/2):]
        val_idx = val_idx[:int(len(val_idx)/2)]


        self.train = Subset(dataset,train_idx)
        self.val = Subset(dataset,val_idx)
        self.test = Subset(dataset,test_idx)
    
    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn = collate_irregular_batch
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
        parser.add_argument('--noise_std', type=float, default=0)
        parser.add_argument('--irregular_rate', type=float, default=1.)
        parser.add_argument('--spline_mode', type=str2bool, default=False,help = "if True, use spline interpolation")
        return parser