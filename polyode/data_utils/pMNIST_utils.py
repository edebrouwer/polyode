from genericpath import exists
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from PIL import Image
import os
from polyode.data_utils.simple_path_utils import get_hermite_spline
from polyode.utils import str2bool
from polyode.data_utils.simple_path_utils import get_hermite_spline, collate_irregular_batch
from polyode.models.spline_cnode import SplineCNODEClass
from torchvision.datasets import MNIST
import tqdm 

from polyode import DATA_DIR

class pMNIST(MNIST):
    """
    This is a Dataset class for a sequence of MNIST digits 
    Each data point consists of num_seqs MNIST images
    Labels are buid by following some rules on these images.
    """
    def __init__(self,train, irregular_rate = 1., spline_mode = False, pre_compute_ode = False, **kwargs):
        super().__init__(train = train, root = os.path.join(DATA_DIR,"MNIST"), download = True) 
        #split the data in num_seqs
        self.data = self.data[:5000]
        self.targets = self.targets[:5000]

        self.sequences = self.data.reshape(self.data.shape[0],-1,1)
        self.sequences = self.sequences / 255 # normalizing the images
        N = self.sequences.shape[0]
        self.xobs = torch.linspace(0,20,784) + 0.02
        self.labels = self.targets.long()
        self.mask = np.random.binomial(1,irregular_rate,size = (N,784)).astype(bool)

        self.sequences[~self.mask] = 0

        self.spline_mode = spline_mode
        if spline_mode:
            self.coeffs = [torch.Tensor(get_hermite_spline(self.xobs,self.sequences[n,:,0],self.mask[n,:])) for n in range(N)]
        
        self.kwargs = kwargs

    def pre_compute_ode_embeddings(self):
        idxs = torch.chunk(torch.arange(self.sequences.shape[0]),20)
        embedding_list = []
        print("Pre-computing ODE Projection embeddings....")
        spline_ode = SplineCNODEClass(**self.kwargs).cuda()
        self.coeffs = torch.stack(self.coeffs)
        for idx in tqdm.tqdm(idxs):
            embedding = spline_ode.integrate_ode(torch.Tensor(self.xobs).cuda(),torch.Tensor(self.sequences[idx]).cuda(),torch.Tensor(self.mask[idx]).cuda(), torch.Tensor(self.coeffs[idx]).cuda())
            embedding_list.append(embedding.cpu())
        self.embeddings = torch.cat(embedding_list)
        self.pre_compute_ode = True

    def __len__(self):
        return self.sequences.shape[0]
    def __getitem__(self,idx):
        if self.spline_mode:
            if self.pre_compute_ode:
                return {"Tobs":self.xobs, "Yobs":self.sequences[idx,:,0], "mask":self.mask[idx], "label":self.labels[idx], "coeffs":self.coeffs[idx], "embeddings":self.embeddings[idx]}
            else:
                return {"Tobs":self.xobs, "Yobs":self.sequences[idx,:,0], "mask":self.mask[idx], "label":self.labels[idx], "coeffs":self.coeffs[idx]}
        else:
            return {"Tobs":self.xobs, "Yobs":self.sequences[idx], "mask":self.mask[idx], "label":self.labels[idx]}

class pMNISTDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, random_seed, num_workers = 4, irregular_rate = 1., spline_mode = False, pre_compute_ode = False, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = random_seed
        self.num_workers = num_workers
        self.irregular_rate = irregular_rate
        self.spline_mode = spline_mode
        self.kwargs = kwargs
        self.pre_compute_ode = pre_compute_ode

    def prepare_data(self, ite_mode = False, treatment2 = None, treatment3 = None):

        dataset = pMNIST(train = True, irregular_rate = self.irregular_rate, spline_mode = self.spline_mode, **self.kwargs)
        
        self.train_idx = np.arange(len(dataset))[:int(0.7*len(dataset))]
        self.val_idx = np.arange(len(dataset))[int(0.7*len(dataset)):]
        
        test_dataset = pMNIST(train = False, irregular_rate = self.irregular_rate, spline_mode = self.spline_mode, **self.kwargs)

        if self.pre_compute_ode:
            dataset.pre_compute_ode_embeddings()
            test_dataset.pre_compute_ode_embeddings()
        
        self.train_batch_size = self.batch_size
        self.val_batch_size = self.batch_size
        self.test_batch_size = self.batch_size

        self.train = Subset(dataset,self.train_idx)
        self.val = Subset(dataset,self.val_idx)
        self.test = test_dataset



    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn = collate_irregular_batch
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn = collate_irregular_batch
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn = collate_irregular_batch
            )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_sequences', type=int, default=4, help = "Number of images per sample")
        parser.add_argument('--irregular_rate', type=float, default=1.)
        parser.add_argument('--spline_mode', type=str2bool, default=False,help = "if True, use spline interpolation")
        parser.add_argument('--pre_compute_ode', type=str2bool, default=False,help = "if True, pre-computes the ODE embedding of the splines")
        return parser


if __name__ == "__main__":
    dataset = pMNIST(train = True)