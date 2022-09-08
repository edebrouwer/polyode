from argparse import ArgumentParser
from legendre.models.cnode import CNODE, CNODEClassification
from legendre.models.cnode_ext import CNODExt, CNODExtClassification
from legendre.models.node_ext import NODExt, NODExtClassification
from legendre.models.spline_cnode import SplineCNODEClass

from legendre.utils import str2bool
import pytorch_lightning as pl
from pytorch_lightning.core.saving import _load_state
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import copy
import os
import torch
import wandb

from legendre.models.node import SequentialODE, SequentialODEClassification
from legendre.models.node_mod import NODE,NODEClassification
from legendre.data_utils.simple_path_utils import SimpleTrajDataModule
from legendre.data_utils.pMNIST_utils import pMNISTDataModule
from legendre.data_utils.character_utils import CharacterTrajDataModule
from legendre.data_utils.mimic_utils import MIMICDataModule
from legendre.data_utils.lorenz_utils import LorenzDataModule
from legendre.models.rnn import RNN, RNNClassification
from legendre.models.hippo import HIPPO, HippoClassification


def get_init_model(init_sweep_id, init_model_cls, irregular_rate, seed, multivariate):
    api = wandb.Api()
    sweep = api.sweep(init_sweep_id)
    runs = sweep.runs
    run = [r for r in runs if (r.config["seed"] == seed) and (
            r.config["irregular_rate"] == irregular_rate) and (r.config.get("multivariate", False) == multivariate)][0]

    fname = [f.name for f in run.files() if "best_model.ckpt" in f.name]
    if len(fname) > 0:
        fname = fname[0]
        run.file(fname).download(replace=True, root=".")
    else:
        print("Wandb model not found, loading local checkpoint")
        fname = os.path.join(".", "checkpoints",
                                 run.name, "best_model.ckpt")

        # output_dim = 1 # hard coded for now
    checkpoint = torch.load(
            fname, map_location=lambda storage, loc: storage)
    checkpoint["hyper_parameters"].pop("callbacks", None)
    checkpoint["hyper_parameters"].pop("logger", None)
    checkpoint["hyper_parameters"].pop("wandb_id_file_path", None)
    init_model = _load_state(init_model_cls, checkpoint) 
    os.remove(fname)
    return init_model

def main(model_cls, init_model_cls, data_cls, args):
    # dataset.prepare_data()
    if init_model_cls is not None:
        init_model = get_init_model(args.init_sweep_id, init_model_cls, args.irregular_rate, args.seed, dict(vars(args)).get("multivariate",False))
        #api = wandb.Api()
        #sweep = api.sweep(args.init_sweep_id)
        #runs = sweep.runs
        #run = [r for r in runs if (r.config["seed"] == args.seed) and (
        #    r.config["irregular_rate"] == args.irregular_rate) and (r.config.get("multivariate", False) == dict(vars(args)).get("multivariate", False))][0]

        #fname = [f.name for f in run.files() if "best_model.ckpt" in f.name]
        #if len(fname) > 0:
        #    fname = fname[0]
        #    run.file(fname).download(replace=True, root=".")
        #else:
        #    print("Wandb model not found, loading local checkpoint")
        #    fname = os.path.join(".", "checkpoints",
        #                         run.name, "best_model.ckpt")

        # output_dim = 1 # hard coded for now
        #checkpoint = torch.load(
        #    fname, map_location=lambda storage, loc: storage)
        #checkpoint["hyper_parameters"].pop("callbacks", None)
        #checkpoint["hyper_parameters"].pop("logger", None)
        #checkpoint["hyper_parameters"].pop("wandb_id_file_path", None)
        #init_model = _load_state(init_model_cls, checkpoint)
        #init_model = init_model_cls._load_model_state(checkpoint)

        #init_model = init_model_cls.load_from_checkpoint(fname, strict=False)

        dataset = data_cls(**vars(args), init_model=init_model)
        model = model_cls(num_dims=dataset.num_dims,
                          init_model=init_model, **vars(args))

    else:
        dataset = data_cls(**vars(args))
        model = model_cls(**vars(args), num_dims=dataset.num_dims)

    logger = WandbLogger(
        name=f"{args.model_type}_Class_{args.data_type}",
        project=f"orthopoly",
        entity="edebrouwer",
        log_model=False
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=logger.experiment.dir,
        monitor='val_loss',
        mode='min',
        verbose=True
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_loss", patience=args.early_stopping)

    trainer = pl.Trainer(gpus=args.gpus, logger=logger, callbacks=[
                         checkpoint_cb, early_stopping_cb], max_epochs=args.max_epochs)
    trainer.fit(model, datamodule=dataset)

    checkpoint_path = checkpoint_cb.best_model_path


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--fold', default=0, type=int,
                        help=' fold number to use')
    parser.add_argument('--gpus', default=1, type=int,
                        help='the number of gpus to use to train the model')
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--early_stopping', default=50, type=int)
    parser.add_argument('--data_type', type=str, default="SimpleTraj")
    parser.add_argument('--model_type', type=str, default="CNODE")
    parser.add_argument('--output_fun', type=str, default=None)
    parser.add_argument('--init_sweep_id', type=str,
                        default="edebrouwer/orthopoly/1m0srjpz")

    partial_args, _ = parser.parse_known_args()

    if partial_args.data_type == "SimpleTraj":
        data_cls = SimpleTrajDataModule
    elif partial_args.data_type == "pMNIST":
        data_cls = pMNISTDataModule
    elif partial_args.data_type == "Character":
        data_cls = CharacterTrajDataModule
    elif partial_args.data_type == "MIMIC":
        data_cls = MIMICDataModule
    elif partial_args.data_type == "Lorenz":
        data_cls = LorenzDataModule

    if partial_args.model_type == "CNODE":
        model_cls = CNODEClassification
        init_model_cls = CNODE
    if partial_args.model_type == "CNODExt":
        model_cls = CNODExtClassification
        init_model_cls = CNODExt
    if partial_args.model_type == "NODExt":
        model_cls = NODExtClassification
        init_model_cls = NODExt
    if partial_args.model_type == "HermiteSpline":
        model_cls = SplineCNODEClass
        init_model_cls = None
    elif partial_args.model_type == "SequentialODE":
        model_cls = SequentialODEClassification
        init_model_cls = SequentialODE
    elif partial_args.model_type == "RNN":
        model_cls = RNNClassification
        init_model_cls = RNN
    elif partial_args.model_type == "Hippo":
        model_cls = HippoClassification
        init_model_cls = HIPPO
    elif partial_args.model_type == "NODE":
        model_cls = NODEClassification
        init_model_cls = NODE

    parser = model_cls.add_model_specific_args(parser)
    parser = data_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()

    main(model_cls, init_model_cls, data_cls, args)
