from argparse import ArgumentParser
from legendre.data_utils.pMNIST_utils import pMNISTDataModule
from legendre.models.cnode import CNODE

from legendre.utils import str2bool
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import copy
import os
import torch

from legendre.models.node import SequentialODE
from legendre.models.cnode import CNODE
from legendre.models.cnode_ext import CNODExt
from legendre.models.node_ext import NODExt
from legendre.models.hippo import HIPPO
from legendre.models.rnn import RNN
from legendre.models.simple_classif import SimpleClassif

from legendre.data_utils.simple_path_utils import SimpleTrajDataModule
from legendre.data_utils.character_utils import CharacterTrajDataModule
from legendre.data_utils.mimic_utils import MIMICDataModule
from legendre.data_utils.lorenz_utils import LorenzDataModule


def main(model_cls, data_cls, args):
    dataset = data_cls(**vars(args))
    dataset.prepare_data()

    output_dim = dataset.num_dims
    model = model_cls(output_dim=output_dim, **vars(args))
    # model.set_classes(num_classes_model=1) #For pretraining, only a single model

    logger = WandbLogger(
        name=f"{args.model_type}_{args.data_type}",
        project=f"orthopoly",
        entity="edebrouwer",
        log_model=False
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=logger.experiment.dir,
        filename='best_model',
        monitor='val_loss',
        mode='min',
        verbose=True
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_loss", patience=args.early_stopping)

    trainer = pl.Trainer(gpus=args.gpus, logger=logger, callbacks=[
                         checkpoint_cb, early_stopping_cb], max_epochs=args.max_epochs, gradient_clip_val=0.5)
    trainer.fit(model, datamodule=dataset)

    checkpoint_path = checkpoint_cb.best_model_path


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--fold', default=0, type=int,
                        help=' fold number to use')
    parser.add_argument('--gpus', default=1, type=int,
                        help='the number of gpus to use to train the model')
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--max_epochs', default=250, type=int)
    parser.add_argument('--early_stopping', default=50, type=int)
    parser.add_argument('--data_type', type=str, default="SimpleTraj")
    parser.add_argument('--model_type', type=str, default="SequentialODE")

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
        model_cls = CNODE
    elif partial_args.model_type == "SequentialODE":
        model_cls = SequentialODE
    elif partial_args.model_type == "CNODExt":
        model_cls = CNODExt
    elif partial_args.model_type == "NODExt":
        model_cls = NODExt
    elif partial_args.model_type == "Hippo":
        model_cls = HIPPO
    elif partial_args.model_type == "RNN":
        model_cls = RNN
    elif partial_args.model_type == "SimpleClassif":
        model_cls = SimpleClassif

    parser = model_cls.add_model_specific_args(parser)
    parser = data_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()

    main(model_cls, data_cls, args)
