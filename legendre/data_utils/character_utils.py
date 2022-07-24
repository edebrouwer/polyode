from genericpath import exists
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from PIL import Image
import os
from legendre.data_utils.simple_path_utils import get_hermite_spline
from legendre.utils import str2bool
from legendre.data_utils.simple_path_utils import get_hermite_spline, collate_irregular_batch
from legendre.models.spline_cnode import SplineCNODEClass
from torchvision.datasets import MNIST
import tqdm 
from sktime.datasets import load_from_tsfile
from scipy.io import loadmat
import torch.nn as nn

from legendre import DATA_DIR

#Check for rosetta

def pad_irregular_characters(df):
    stacked_list = []
    max_len_stacked = 0
    for idx in range(df.shape[0]):
        max_len = np.max(np.array([len(df.iloc[idx]["dim_0"]), len(df.iloc[idx]["dim_1"]), len(df.iloc[idx]["dim_2"])])).astype(int)
        stacked_vec = np.stack([np.pad(df.iloc[idx]["dim_0"],(0,max_len-len(df.iloc[idx]["dim_0"]))), np.pad(df.iloc[idx]["dim_1"],(0,max_len-len(df.iloc[idx]["dim_1"]))), np.pad(df.iloc[idx]["dim_2"],(0,max_len-len(df.iloc[idx]["dim_2"])))],-1)
        if len(stacked_vec)>max_len_stacked:
            max_len_stacked = len(stacked_vec)
        stacked_list.append(stacked_vec)

    x = np.stack([np.pad(stack,((0,max_len_stacked-len(stack)),(0,0))) for stack in stacked_list])
    return x

def to_tensor(df):
    return np.stack([np.stack([df.iloc[idx]["dim_0"], df.iloc[idx]["dim_1"], df.iloc[idx]["dim_2"]],-1) for idx in range(df.shape[0])])

def mixed_loader():
    path = os.path.join(DATA_DIR,"CharacterTraj","mixoutALL_shifted.mat")
    file = loadmat(path,squeeze_me = True)

    y = file["consts"]["charlabels"].item()
    x = file["mixout"]

    padded = nn.utils.rnn.pad_sequence([torch.Tensor(x_.T) for x_ in x],batch_first = True).numpy()
    min_len = np.min([x_.shape[1] for x_ in x])
    cropped = np.stack([x_.T[:min_len] for x_ in x])
    return cropped, y-1

class CharacterTraj(Dataset):
    """
    This is a Dataset class for a sequence of MNIST digits 
    Each data point consists of num_seqs MNIST images
    Labels are buid by following some rules on these images.
    """
    def __init__(self,train, irregular_rate = 1., spline_mode = False, pre_compute_ode = False, **kwargs):
        super().__init__()
        
        #if train:
        #    path = os.path.join(DATA_DIR,"CharacterTraj","CharacterTrajectories_TRAIN.ts") 
        #else:
        #    path = os.path.join(DATA_DIR,"CharacterTraj","CharacterTrajectories_TEST.ts") 


        #df_x, df_y = load_from_tsfile(path)
        
        #self.data = to_tensor(df_x)
        #self.data = pad_irregular_characters(df_x)

        self.data, self.targets = mixed_loader()
        
        #map_dict = dict(zip(np.unique(df_y),np.arange(len(np.unique(df_y)))))
        #self.targets = np.vectorize(map_dict.get)(df_y)
        # CURRENTLY ONLY SELECTING THE FIRST DIMENSION
        self.sequences = self.data[:,::2,0][...,None]
        #self.sequences = self.data[:,::2,:]
        t_len = self.sequences.shape[1]
        N = self.sequences.shape[0]
        self.xobs = torch.linspace(0,10,t_len) + 0.1
        self.labels = self.targets.astype(int)
        self.mask = np.random.binomial(1,irregular_rate,size = (N,t_len)).astype(bool)
        self.mask[:,-1] = True #last element always observed

        self.sequences[~self.mask] = 0

        self.spline_mode = spline_mode
        if spline_mode:
            self.coeffs = [torch.Tensor(get_hermite_spline(self.xobs,self.sequences[n,:,0],self.mask[n,:])) for n in range(N)]

        self.kwargs = kwargs
        self.pre_compute_ode = pre_compute_ode



        if kwargs.get("bridge_ode",False):
            self.bridge_ode = True
            ids_vec = []
            ys_vec = []
            ts_vec = []
            mask_vec = []
            max_len = 0
            for idx in range(N):
                id0s = np.where(self.mask[idx]==1)[0][::2]
                id1s = id0s[1:]
                id0s = id0s[:-1]
                idmids = np.where(self.mask[idx]==1)[0][1::2][:len(id1s)]
                t0s = self.xobs[id0s]
                t1s = self.xobs[id1s]
                tmids = self.xobs[idmids]
                y0s = self.sequences[idx,id0s,:]
                y1s = self.sequences[idx,id1s,:]
                ymids = self.sequences[idx,idmids,:]
                ids = np.concatenate((id0s[:,None],id1s[:,None],idmids[:,None]),axis = 1)
                ts = np.concatenate((t0s[:,None],t1s[:,None],tmids[:,None]),axis = 1)
                ys = np.concatenate((y0s,y1s,ymids),axis = 1)
                ids_vec.append(torch.Tensor(ids))
                ts_vec.append(torch.Tensor(ts))
                ys_vec.append(torch.Tensor(ys))
                mask_vec.append(torch.ones(len(ts)))
                if len(ts)>max_len:
                    max_len = len(ts)
            self.ts = torch.nn.utils.rnn.pad_sequence(ts_vec,batch_first = True).numpy()
            self.ids = torch.nn.utils.rnn.pad_sequence(ids_vec,batch_first = True).numpy()
            self.ys = torch.nn.utils.rnn.pad_sequence(ys_vec,batch_first = True).numpy()
            self.mask_ids = torch.nn.utils.rnn.pad_sequence(mask_vec,batch_first = True).numpy()
        else:
            self.bridge_ode = False
            


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
                return_dict =  {"Tobs":self.xobs, "Yobs":self.sequences[idx,:,0], "mask":self.mask[idx], "label":self.labels[idx], "coeffs":self.coeffs[idx], "embeddings":self.embeddings[idx]}
            else:
                return_dict =  {"Tobs":self.xobs, "Yobs":self.sequences[idx,:,0], "mask":self.mask[idx], "label":self.labels[idx], "coeffs":self.coeffs[idx]}
        else:
            return_dict =  {"Tobs":self.xobs, "Yobs":self.sequences[idx], "mask":self.mask[idx], "label":self.labels[idx]}

        if self.bridge_ode:
            extra_dict = {"ts":self.ts[idx],"ids":self.ids[idx],"ys":self.ys[idx],"mask_ids":self.mask_ids[idx]}
            return_dict.update(extra_dict)
        return return_dict



class CharacterTrajDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, seed, num_workers = 4, irregular_rate = 1., spline_mode = False, pre_compute_ode = False, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.irregular_rate = irregular_rate
        self.spline_mode = spline_mode
        self.kwargs = kwargs
        self.pre_compute_ode = pre_compute_ode
        self.seed = seed

    def prepare_data(self):

        dataset = CharacterTraj(train = True, irregular_rate = self.irregular_rate, spline_mode = self.spline_mode, **self.kwargs)
        
        np.random.seed(seed= self.seed)
        idx_full = np.random.permutation(len(dataset))
        self.train_idx = idx_full[:int(0.7*len(dataset))]
        self.val_idx = idx_full[int(0.7*len(dataset)):]
        
        test_dataset = CharacterTraj(train = False, irregular_rate = self.irregular_rate, spline_mode = self.spline_mode, **self.kwargs)

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
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_sequences', type=int, default=4, help = "Number of images per sample")
        parser.add_argument('--irregular_rate', type=float, default=1.)
        parser.add_argument('--spline_mode', type=str2bool, default=False,help = "if True, use spline interpolation")
        parser.add_argument('--pre_compute_ode', type=str2bool, default=False,help = "if True, pre-computes the ODE embedding of the splines")
        return parser


if __name__ == "__main__":
    dataset = CharacterTraj(train = True)