import numpy as np
import torch
import pytorch_lightning as pl

from scipy.integrate import odeint
from torch.utils.data import Dataset, DataLoader, Subset
from legendre.utils import str2bool

def generate_path(phase,Nt,Nobs):
    """
    Nt : Number of time points to simulate
    Nobs : Number of observations to select
    phase : phase to add to this particular time series
    """
    x = np.linspace(0,10,Nt)
    y = np.sin(x + phase) * np.cos(3*(x+phase))
    xobs = x[1:-1:Nt//Nobs]
    yobs = y[1:-1:Nt//Nobs]
    return x,y, xobs, yobs

def generate_dataset(N,Nt,Nobs):
    Xobs = []
    Yobs = []
    for n in range(N):
        phase = 2*np.random.randn()*np.pi
        x,y,xobs, yobs = generate_path(phase,Nt,Nobs)
        
        Xobs.append(xobs)
        Yobs.append(yobs)
    Xobs = np.stack(Xobs)
    Yobs = np.stack(Yobs)
    return Xobs, Yobs


class SimpleTrajDataset(Dataset):
    def __init__(self,N,Nt,Nobs):
        super().__init__()
        self.N = N
        self.Nt = Nt
        self.Nobs = Nobs

        self.Tobs, self.Yobs = generate_dataset(N,Nt,Nobs)

    def __len__(self):
        return self.N
    
    def __getitem__(self,idx):
        return self.Tobs[idx], self.Yobs[idx]


class SimpleTrajDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, seed, N,  noise_std,  num_workers = 4,  **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True
        self.noise_std = noise_std
        self.N = N

    def prepare_data(self):

        dataset = SimpleTrajDataset(N = self.N, noise_std =  self.noise_std, seed = self.seed)       
        
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
            pin_memory=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
            )
    
    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--N_ts', type=int, default=1000)
        parser.add_argument('--noise_std', type=float, default=0)
        return parser