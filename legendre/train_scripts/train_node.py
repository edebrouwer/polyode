from argparse import ArgumentParser

from legendre.utils import str2bool
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import copy
import os
import torch

from legendre.models.node import SequentialODE
from legendre.data_utils.simple_path_utils import SimpleTrajDataModule

def main(model_cls, data_cls, args):
    dataset = data_cls(**vars(args))
    dataset.prepare_data()

    model = model_cls(**vars(args))
    #model.set_classes(num_classes_model=1) #For pretraining, only a single model
    
    logger = WandbLogger(
        name=f"CF_{args.data_type}",
        project=f"counterfactuals",
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

    partial_args, _ = parser.parse_known_args()

    model_cls = SequentialODE 
    
    if partial_args.data_type == "SimpleTraj":
        data_cls = SimpleTrajDataModule

    parser = model_cls.add_model_specific_args(parser)
    parser = data_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()


    main(model_cls, data_cls, args)
