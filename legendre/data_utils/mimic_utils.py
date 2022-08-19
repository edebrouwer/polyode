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


class DictDist():
    def __init__(self, dict_of_rvs): self.dict_of_rvs = dict_of_rvs

    def rvs(self, n):
        a = {k: v.rvs(n) for k, v in self.dict_of_rvs.items()}
        out = []
        for i in range(n):
            out.append({k: vs[i] for k, vs in a.items()})
        return out


class Choice():
    def __init__(self, options): self.options = options
    def rvs(self, n): return [self.options[i]
                              for i in ss.randint(0, len(self.options)).rvs(n)]


def simple_imputer(df, ID_COLS):
    idx = pd.IndexSlice
    df = df.copy()
    if len(df.columns.names) > 2:
        df.columns = df.columns.droplevel(('label', 'LEVEL1', 'LEVEL2'))

    df_out = df.loc[:, idx[:, ['mean', 'count']]]
    icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).mean()

    df_out.loc[:, idx[:, 'mean']] = df_out.loc[:, idx[:, 'mean']].groupby(
        ID_COLS).fillna(0).groupby(ID_COLS).fillna(icustay_means).fillna(0)

    df_out.loc[:, idx[:, 'count']] = (
        df.loc[:, idx[:, 'count']] > 0).astype(float)
    df_out.rename(columns={'count': 'mask'},
                  level='Aggregation Function', inplace=True)

    is_absent = (1 - df_out.loc[:, idx[:, 'mask']])
    hours_of_absence = is_absent.cumsum()
    time_since_measured = hours_of_absence - \
        hours_of_absence[is_absent == 0].fillna(method='ffill')
    time_since_measured.rename(
        columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)

    df_out = pd.concat((df_out, time_since_measured), axis=1)
    df_out.loc[:, idx[:, 'time_since_measured']] = df_out.loc[:,
                                                              idx[:, 'time_since_measured']].fillna(100)

    df_out.sort_index(axis=1, inplace=True)
    return df_out


def to_3D_tensor(df):
    idx = pd.IndexSlice
    return np.dstack((df.loc[idx[:, :, :, i], :].values for i in sorted(set(df.index.get_level_values('hours_in')))))


def prepare_dataloader(df, Ys, batch_size, shuffle=True):
    """
    dfs = (df_train, df_dev, df_test).
    df_* = (subject, hadm, icustay, hours_in) X (level2, agg fn \ni {mask, mean, time})
    Ys_series = (subject, hadm, icustay) => label.
    """
    X = torch.from_numpy(to_3D_tensor(df).astype(np.float32))
    label = torch.from_numpy(Ys.values.astype(np.int64))
    dataset = torch.utils.data.TensorDataset(X, label)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def process_mimic_dataset():

    DATA_FILEPATH = os.path.join(
        DATA_DIR, 'physionet.org', 'all_hourly_data.h5')
    GAP_TIME = 6  # In hours
    WINDOW_SIZE = 24  # In hours
    SEED = 1
    ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_full_lvl2 = pd.read_hdf(DATA_FILEPATH, 'vitals_labs')
    statics = pd.read_hdf(DATA_FILEPATH, 'patients')

    Ys = statics[statics.max_hours > WINDOW_SIZE +
                 GAP_TIME][['mort_hosp', 'mort_icu', 'los_icu']]
    Ys['los_3'] = Ys['los_icu'] > 3
    Ys['los_7'] = Ys['los_icu'] > 7
    Ys.drop(columns=['los_icu'], inplace=True)
    Ys.astype(float)

    lvl2 = data_full_lvl2[
        (data_full_lvl2.index.get_level_values('icustay_id').isin(set(Ys.index.get_level_values('icustay_id')))) &
        (data_full_lvl2.index.get_level_values('hours_in') < WINDOW_SIZE)
    ]

    # raw.columns = raw.columns.droplevel(level=['label', 'LEVEL1', 'LEVEL2'])

    train_frac, dev_frac, test_frac = 0.7, 0.1, 0.2
    lvl2_subj_idx, Ys_subj_idx = [
        df.index.get_level_values('subject_id') for df in (lvl2, Ys)]
    lvl2_subjects = set(lvl2_subj_idx)
    assert lvl2_subjects == set(Ys_subj_idx), "Subject ID pools differ!"
    # assert lvl2_subjects == set(raw_subj_idx), "Subject ID pools differ!"

    np.random.seed(SEED)
    subjects, N = np.random.permutation(
        list(lvl2_subjects)), len(lvl2_subjects)
    N_train, N_dev, N_test = int(
        train_frac * N), int(dev_frac * N), int(test_frac * N)
    train_subj = subjects[:N_train]
    dev_subj = subjects[N_train:N_train + N_dev]
    test_subj = subjects[N_train+N_dev:]

    [(lvl2_train, lvl2_dev, lvl2_test), (Ys_train, Ys_dev, Ys_test)] = [
        [df[df.index.get_level_values('subject_id').isin(s)]
         for s in (train_subj, dev_subj, test_subj)]
        for df in (lvl2, Ys)
    ]

    idx = pd.IndexSlice
    lvl2_means, lvl2_stds = lvl2_train.loc[:, idx[:, 'mean']].mean(
        axis=0), lvl2_train.loc[:, idx[:, 'mean']].std(axis=0)

    lvl2_train.loc[:, idx[:, 'mean']] = (
        lvl2_train.loc[:, idx[:, 'mean']] - lvl2_means)/lvl2_stds
    lvl2_dev.loc[:, idx[:, 'mean']] = (
        lvl2_dev.loc[:, idx[:, 'mean']] - lvl2_means)/lvl2_stds
    lvl2_test.loc[:, idx[:, 'mean']] = (
        lvl2_test.loc[:, idx[:, 'mean']] - lvl2_means)/lvl2_stds

    lvl2_train, lvl2_dev, lvl2_test = [
        simple_imputer(df, ID_COLS) for df in (lvl2_train, lvl2_dev, lvl2_test)
    ]
    lvl2_flat_train, lvl2_flat_dev, lvl2_flat_test = [
        df.pivot_table(index=['subject_id', 'hadm_id', 'icustay_id'], columns=['hours_in']) for df in (
            lvl2_train, lvl2_dev, lvl2_test
        )
    ]

    for df in lvl2_train, lvl2_dev, lvl2_test:
        assert not df.isnull().any().any()

    Ys = statics[statics.max_hours > WINDOW_SIZE +
                 GAP_TIME][['mort_hosp', 'mort_icu', 'los_icu']]
    Ys['los_3'] = Ys['los_icu'] > 3
    Ys['los_7'] = Ys['los_icu'] > 7
    Ys.drop(columns=['los_icu'], inplace=True)
    Ys.astype(float)
    [(Ys_train, Ys_dev, Ys_test)] = [
        [df[df.index.get_level_values('subject_id').isin(s)]
         for s in (train_subj, dev_subj, test_subj)]
        for df in (Ys,)
    ]

    lvl2_train.to_pickle(os.path.join(
        DATA_DIR, 'physionet.org', "processed", "lvl2_train.pkl"))
    lvl2_dev.to_pickle(os.path.join(
        DATA_DIR, 'physionet.org', "processed", "lvl2_dev.pkl"))
    lvl2_test.to_pickle(os.path.join(
        DATA_DIR, 'physionet.org', "processed", "lvl2_test.pkl"))

    Ys_train.to_pickle(os.path.join(
        DATA_DIR, 'physionet.org', "processed", "Ys_train.pkl"))
    Ys_dev.to_pickle(os.path.join(
        DATA_DIR, 'physionet.org', "processed", "Ys_dev.pkl"))
    Ys_test.to_pickle(os.path.join(
        DATA_DIR, 'physionet.org', "processed", "Ys_test.pkl"))


class MIMICDataset(Dataset):
    def __init__(self, fold_type="train", spline_mode=False, pre_compute_ode=False, label_type="mort_hosp", irregular_rate=1, **kwargs):
        super().__init__()
        df_traj = pd.read_pickle(os.path.join(
            DATA_DIR, 'physionet.org', "processed", f"lvl2_{fold_type}.pkl"))
        df_Y = pd.read_pickle(os.path.join(
            DATA_DIR, 'physionet.org', "processed", f"Ys_{fold_type}.pkl"))

        #TODO: add creatinine. Check feature importance of mortality.
        cols = ["heart rate", "mean blood pressure", "diastolic blood pressure", "oxygen saturation","respiratory rate"]
        self.X = torch.from_numpy(to_3D_tensor(
            df_traj[cols]).astype(np.float32))
        self.labels = torch.from_numpy(
            df_Y[label_type].values.astype(np.int64))

        if fold_type == "train":
            self.X = self.X[:5000]
            self.labels = self.labels[:5000]

        self.xobs = torch.linspace(0, 10, 24) + 0.1
        self.mask = self.X[:, 0::3].permute(0, 2, 1)
        self.sequences = self.X[:, 1::3].permute(0, 2, 1)

        N = self.X.shape[0]
        # only taking observations where all dimensions are observed
        self.mask = self.mask.all(-1).float()
        self.add_mask = np.random.binomial(
            1, irregular_rate, size=(N, len(self.xobs))).astype(np.float32)
        self.mask = self.mask * self.add_mask
        self.sequences[self.mask == 0] = 0

        self.spline_mode = spline_mode
        if spline_mode:
            self.coeffs = [torch.stack([torch.Tensor(get_hermite_spline(
                self.xobs, self.sequences[n, :, dim], self.mask[n, :])) for dim in range(self.sequences.shape[-1])], -1) for n in range(N)]

        self.kwargs = kwargs
        self.pre_compute_ode = pre_compute_ode

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
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.pre_compute_ode:
            return_dict = {"Tobs": self.xobs, "Yobs": self.sequences[idx], "mask": self.mask[idx],
                           "label": self.labels[idx], "embeddings": self.embeddings[idx]}
        elif self.spline_mode:
            return_dict = {"Tobs": self.xobs, "Yobs": self.sequences[idx],
                           "mask": self.mask[idx], "label": self.labels[idx], "coeffs": self.coeffs[idx]}
        else:
            return_dict = {
                "Tobs": self.xobs, "Yobs": self.sequences[idx], "mask": self.mask[idx], "label": self.labels[idx]}
        return return_dict


class MIMICDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, seed=42, num_workers=4, irregular_rate=1., spline_mode=False, pre_compute_ode=False, multivariate=False, **kwargs):

        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.irregular_rate = irregular_rate
        self.spline_mode = spline_mode
        self.kwargs = kwargs
        self.pre_compute_ode = pre_compute_ode
        self.seed = seed
        self.num_dims = 5
        self.test_only = False

    def set_test_only(self):
        self.test_only = True

    def prepare_data(self):

        train_dataset = MIMICDataset(fold_type="train", irregular_rate=self.irregular_rate,
                                     spline_mode=self.spline_mode, **self.kwargs)
        val_dataset = MIMICDataset(fold_type="dev", irregular_rate=self.irregular_rate,
                                   spline_mode=self.spline_mode, **self.kwargs)

        test_dataset = MIMICDataset(
            fold_type="test", irregular_rate=self.irregular_rate, spline_mode=self.spline_mode, **self.kwargs)

        if self.pre_compute_ode:
            if not self.test_only:
                train_dataset.pre_compute_ode_embeddings(**self.kwargs)
                val_dataset.pre_compute_ode_embeddings(**self.kwargs)
            test_dataset.pre_compute_ode_embeddings(**self.kwargs)

        self.train_batch_size = self.batch_size
        self.val_batch_size = self.batch_size
        self.test_batch_size = self.batch_size

        self.train = train_dataset
        self.val = val_dataset
        self.test = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_irregular_batch
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_irregular_batch
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.test_batch_size,
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
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--num_sequences', type=int,
                            default=4, help="Number of images per sample")
        parser.add_argument('--irregular_rate', type=float, default=1.)
        parser.add_argument('--spline_mode', type=str2bool,
                            default=False, help="if True, use spline interpolation")
        parser.add_argument('--pre_compute_ode', type=str2bool, default=False,
                            help="if True, pre-computes the ODE embedding of the splines")
        return parser


if __name__ == "__main__":
    # process_mimic_dataset()  # run this to create the processed dataframes
    ds = MIMICDataset()
