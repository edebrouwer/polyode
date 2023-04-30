from legendre import DATA_DIR
import os
import numpy as np
import torch
import pandas as pd
import tqdm
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset
from legendre.utils import str2bool
from legendre.data_utils.simple_path_utils import get_hermite_spline, collate_irregular_batch
from legendre.models.spline_cnode import SplineCNODEClass

from numpy.random import default_rng


def collate_activity_batch(batch):
    """_summary_

    Args:
        batch (_type_): _description_

    Returns:
        _type_: _description_
    """
    T_obs = [b["Tobs"] for b in batch]
    Y_obs = torch.stack([b["Yobs"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])

    obs_mask = masks.bool().any(0).any(1)
    Y_obs = Y_obs[:,obs_mask,:]
    masks = masks[:,obs_mask,:]
    unique_times = T_obs[0][obs_mask]
    #used_times = masks.sum(0) > 0
    #assert((masks.sum(0) > 0).all())
    # unique_times = unique_times[used_times] #check if some times are never used.

    #Y_obs = Y_obs[:, used_times]
    #masks = masks[:, used_times]

    if "coeffs" in batch[0]:
        coeffs = np.stack([b["coeffs"] for b in batch])
        return_tuple = (torch.Tensor(unique_times), torch.Tensor(
            Y_obs), torch.Tensor(masks), labels, torch.Tensor(coeffs))
    elif "embeddings" in batch[0]:
        embeddings = np.stack([b["embeddings"] for b in batch])
        return_tuple = (torch.Tensor(unique_times), torch.Tensor(
            Y_obs), torch.Tensor(masks), labels, torch.Tensor(embeddings))
    else:
        return_tuple = (unique_times,
            Y_obs, masks, labels, None)

    if "ts" in batch[0]:
        ts = np.stack([b["ts"] for b in batch])
        ys = np.stack([b["ys"] for b in batch])
        ids = np.stack([b["ids"] for b in batch])
        mask_ids = np.stack([b["mask_ids"] for b in batch])
        return_tuple = return_tuple + \
            (torch.Tensor(ids), torch.Tensor(ts),
             torch.Tensor(ys), torch.Tensor(mask_ids))

    if "Y_future" in batch[0]:
        Y_future = np.stack([b["Y_future"] for b in batch])
        Y_past = np.stack([b["Y_past"] for b in batch])
        mask_future = np.stack([b["mask_future"] for b in batch])
        mask_past = np.stack([b["mask_past"] for b in batch])
        return_tuple = return_tuple + (torch.Tensor(Y_past), torch.Tensor(
            Y_future), torch.Tensor(mask_past), torch.Tensor(mask_future))

    return return_tuple


class ActivityDataset(Dataset):
    def __init__(self, spline_mode=False, pre_compute_ode=False, irregular_rate=1, forecast_mode = False, **kwargs):
        super().__init__()

        self.sequences = torch.Tensor(np.load(os.path.join(DATA_DIR, "Activity", "tensor.npy")))
        self.mask = torch.Tensor(np.load(os.path.join(DATA_DIR, "Activity", "mask.npy")))
        self.labels = torch.Tensor(np.load(os.path.join(DATA_DIR, "Activity", "label.npy")))
        mean_seq = torch.nanmean(self.sequences,dim=(0,1),keepdim=True)
        std_seq = np.nanstd(self.sequences,axis=(0,1),keepdims=True)

        self.sequences = (self.sequences - mean_seq)/std_seq

        self.xobs = torch.linspace(0, 20, self.sequences.shape[1]) + 0.1

        self.Nobs = len(self.xobs)
        N = self.sequences.shape[0]
        # only taking observations where all dimensions are observed
        #self.mask = self.mask.all(-1).float()

        self.sequences[self.mask == 0] = 0

        self.spline_mode = spline_mode
        if spline_mode:
            self.coeffs = [torch.stack([torch.Tensor(get_hermite_spline(
                self.xobs, self.sequences[n, :, dim], self.mask[n, :, dim])) for dim in range(self.sequences.shape[-1])], -1) for n in range(N)]

        self.kwargs = kwargs
        self.pre_compute_ode = pre_compute_ode
        self.forecast_mode = forecast_mode

    def pre_compute_ode_embeddings(self, **kwargs):
        idxs = torch.chunk(torch.arange(self.sequences.shape[0]), 20)
        embedding_list = []
        print("Pre-computing ODE Projection embeddings....")
        if "init_model" in kwargs:
            model = kwargs["init_model"]
            model.eval()
            model.cuda()
            for idx in tqdm.tqdm(idxs):
                embedding = model.get_embedding(torch.Tensor(self.xobs).cuda(), torch.Tensor(
                    self.sequences[idx]).cuda(), torch.Tensor(self.mask[idx]).cuda())
                embedding_list.append(embedding.cpu())
        else:
            if "num_dims" not in self.kwargs:
                self.kwargs["num_dims"] = self.num_dims
            spline_ode = SplineCNODEClass(
                **self.kwargs).cuda()
            self.coeffs = torch.stack(self.coeffs)
            for idx in tqdm.tqdm(idxs):
                embedding = spline_ode.integrate_ode(torch.Tensor(self.xobs).cuda(), torch.Tensor(
                    self.sequences[idx]).cuda(), torch.Tensor(self.mask[idx]).cuda(), torch.Tensor(self.coeffs[idx]).cuda())
                embedding_list.append(embedding.cpu())
        self.embeddings = torch.cat(embedding_list)
        self.pre_compute_ode = True

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        if self.pre_compute_ode:
            return_dict = {"Tobs": self.xobs, "Yobs": self.sequences[idx], "mask": self.mask[idx],
                           "label": self.labels[idx], "embeddings": self.embeddings[idx]}
        elif self.spline_mode:
            return_dict = {"Tobs": self.xobs, "Yobs": self.sequences[idx],
                           "mask": self.mask[idx], "label": self.labels[idx], "coeffs": self.coeffs[idx]}
        elif self.forecast_mode:
            Y_future = self.sequences[idx].clone()
            mask_future = self.mask[idx].clone()
            mask_past = self.mask[idx].clone()
            Y_past = self.sequences[idx].clone()
            Y_future[int(0.8*self.Nobs):] = 0
            Y_past[:int(0.8*self.Nobs)] = 0
            mask_future[int(0.8*self.Nobs):] = 0
            mask_past[:int(0.8*self.Nobs)] = 0
            return {"Tobs": self.xobs, "Yobs": self.sequences[idx], "label": self.labels[idx], "mask": self.mask[idx], "Y_future": Y_future, "mask_future": mask_future, "Y_past": Y_past, "mask_past": mask_past}
        else:
            return_dict = {
                "Tobs": self.xobs, "Yobs": self.sequences[idx], "mask": self.mask[idx], "label": self.labels[idx]}
        return return_dict


class ActivityDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, seed=42, num_workers=4, irregular_rate=1., spline_mode=False, pre_compute_ode=False, regression_mode=False, forecast_mode = False, **kwargs):

        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.irregular_rate = irregular_rate
        self.spline_mode = spline_mode
        self.kwargs = kwargs
        self.pre_compute_ode = pre_compute_ode
        self.seed = seed
        self.num_dims = 12
        self.test_only = False
        self.regression_mode = regression_mode
        self.forecast_mode = forecast_mode

    def set_test_only(self):
        self.test_only = True

    def prepare_data(self):

        dataset = ActivityDataset(irregular_rate=self.irregular_rate,
                                     spline_mode=self.spline_mode, forecast_mode = self.forecast_mode,**self.kwargs)

        if self.pre_compute_ode:
            dataset.pre_compute_ode_embeddings(**self.kwargs)

        self.train_batch_size = self.batch_size
        self.val_batch_size = self.batch_size
        self.test_batch_size = self.batch_size

        rng = np.random.default_rng(seed = self.seed)
        idxs = np.arange(len(dataset))
        rng.shuffle(idxs)

        train_idx = idxs[:int(0.8*len(dataset))]
        val_idx = idxs[int(0.8*len(dataset)):int(0.9*len(dataset))]
        test_idx = idxs[int(0.9*len(dataset)):]

        self.train = Subset(dataset, train_idx)
        self.val = Subset(dataset, val_idx)
        self.test = Subset(dataset, test_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_activity_batch
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_activity_batch
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_activity_batch
        )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_sequences', type=int,
                            default=4, help="Number of images per sample")
        parser.add_argument('--irregular_rate', type=float, default=1.)
        parser.add_argument('--spline_mode', type=str2bool,
                            default=False, help="if True, use spline interpolation")
        parser.add_argument('--pre_compute_ode', type=str2bool, default=False,
                            help="if True, pre-computes the ODE embedding of the splines")
        parser.add_argument('--regression_mode', type=str2bool, default=False,
                            help="if True, splits the sequence into a past and future part")
        parser.add_argument('--forecast_mode', type=str2bool, default=False,
                            help="if True, splits the sequence into a past and future part")
        return parser


if __name__ == "__main__":
    # run this to create the processed dataframes
    data_module = ActivityDataModule(seed=421,num_workers = 0)
    data_module.prepare_data()
    dl = data_module.train_dataloader()
    batch = next(iter(dl))
    import ipdb; ipdb.set_trace()
