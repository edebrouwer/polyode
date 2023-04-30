from argparse import ArgumentParser
from polyode.data_utils.pMNIST_utils import pMNISTDataModule
from polyode.models.cnode import CNODE

from polyode.utils import str2bool
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import copy
import os
import torch

from polyode.models.node import SequentialODE
from polyode.models.cnode import CNODE
from polyode.models.cnode_ext import CNODExt
from polyode.models.node_ext import NODExt
from polyode.models.node_mod import NODE
from polyode.models.hippo import HIPPO
from polyode.models.rnn import RNN
from polyode.models.simple_classif import SimpleClassif

from polyode.data_utils.simple_path_utils import SimpleTrajDataModule
from polyode.data_utils.character_utils import CharacterTrajDataModule
from polyode.data_utils.mimic_utils import MIMICDataModule
from polyode.data_utils.lorenz_utils import LorenzDataModule
from polyode.data_utils.activity_utils import ActivityDataModule


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

    #tester = pl.Trainer(gpus=args.gpus, logger=logger,resume_from_checkpoint=checkpoint_path)
    #tester.test(model, dataloaders=dataset.test_dataloader())


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
    elif partial_args.data_type == "Activity":
        data_cls = ActivityDataModule

    if partial_args.model_type == "CNODE":
        model_cls = CNODE
    elif partial_args.model_type == "SequentialODE":
        model_cls = SequentialODE
    elif partial_args.model_type == "NODE":
        model_cls = NODE
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
