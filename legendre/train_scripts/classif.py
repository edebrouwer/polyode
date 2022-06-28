from argparse import ArgumentParser
from legendre.models.cnode import CNODE, CNODEClassification

from legendre.utils import str2bool
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import copy
import os
import torch
import wandb

from legendre.models.node import SequentialODE, SequentialODEClassification
from legendre.data_utils.simple_path_utils import SimpleTrajDataModule

def main(model_cls, init_model_cls, data_cls, args):
    dataset = data_cls(**vars(args))
    dataset.prepare_data()

    api = wandb.Api()
    run = api.run(args.init_run_id)
    fname = [f.name for f in run.files() if "ckpt" in f.name][0]
    run.file(fname).download(replace = True, root = ".")

    output_dim = 1 # hard coded for now
    init_model = init_model_cls.load_from_checkpoint(fname)
    os.remove(fname)
    model = model_cls(output_dim = 1, init_model = init_model, **vars(args))

    logger = WandbLogger(
        name=f"NODE_Class_{args.data_type}",
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
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=args.early_stopping)

    trainer = pl.Trainer(gpus = args.gpus, logger = logger, callbacks = [checkpoint_cb, early_stopping_cb], max_epochs = args.max_epochs)
    trainer.fit(model, datamodule = dataset)

    checkpoint_path = checkpoint_cb.best_model_path
    

if __name__=="__main__":
    
    parser = ArgumentParser()

    parser.add_argument('--fold', default=0, type=int, help=' fold number to use')
    parser.add_argument('--gpus', default=1, type=int, help='the number of gpus to use to train the model')
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--max_epochs', default=500, type=int)
    parser.add_argument('--early_stopping', default=20, type=int)
    parser.add_argument('--data_type', type = str, default = "SimpleTraj")
    parser.add_argument('--model_type', type = str, default = "CNODEClass")
    parser.add_argument('--init_run_id', type = str, default = "edebrouwer/orthopoly/n2n275j9")

    partial_args, _ = parser.parse_known_args()

    
    if partial_args.data_type == "SimpleTraj":
        data_cls = SimpleTrajDataModule

    if partial_args.model_type == "CNODEClass":
        model_cls = CNODEClassification
        init_model_cls = CNODE
    elif partial_args.model_type == "SequentialODE":
        model_cls = SequentialODEClassification
        init_model_cls = SequentialODE

    parser = model_cls.add_model_specific_args(parser)
    parser = data_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()

    main(model_cls, init_model_cls, data_cls, args)
