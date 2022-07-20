import torch
import pytorch_lightning as pl
import torch.nn as nn

from sklearn.metrics import roc_auc_score, accuracy_score
from legendre.utils import str2bool

class RNN(pl.LightningModule):
    def __init__(self,lr,
        hidden_dim,
        weight_decay,
        Nc,
        **kwargs
    ):

        super().__init__()
        self.save_hyperparameters()

        self.Nc = Nc
        self.rnn_model = torch.nn.GRU(input_size=1, hidden_size = Nc,num_layers = 1, batch_first = True)

        if self.hparams["data_type"]=="pMNIST":
            self.loss_class = torch.nn.CrossEntropyLoss()
            output_dim = 10
        elif self.hparams["data_type"]=="Character":
            self.loss_class = torch.nn.CrossEntropyLoss()
            output_dim = 20
        else:
            self.loss_class = torch.nn.BCEWithLogitsLoss()
            output_dim = 1

        self.classif_model = nn.Sequential(nn.Linear(Nc,hidden_dim),nn.ReLU(), nn.Linear(hidden_dim,output_dim))
    

    def forward(self,times,Y,mask, eval_mode = False):
        _, embeddings = self.rnn_model(Y)
        embeddings = embeddings.permute((1,2,0)).reshape(Y.shape[0],-1)
        preds = self.classif_model(embeddings)
        return preds

    def training_step(self,batch, batch_idx):
        times, Y, mask, label = batch
        preds = self(times,Y,mask)
        if preds.shape[-1]==1:
            preds = preds[:,0]
            loss = self.loss_class(preds.double(),label)
        else:
            loss = self.loss_class(preds.double(),label.long())
        self.log("train_loss",loss,on_epoch=True)
        return {"loss":loss, "Y":Y, "preds":preds, "T":times, "labels":label}


    def training_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        if (self.hparams["data_type"]=="pMNIST") or (self.hparams["data_type"]=="Character"):
            preds = torch.nn.functional.softmax(preds,dim=-1).argmax(-1)
            accuracy = accuracy_score(labels.long().cpu().numpy(),preds.cpu().numpy())
            self.log("train_acc",accuracy,on_epoch=True)
        else:
            auc = roc_auc_score(labels.cpu().numpy(),preds.cpu().numpy())
            self.log("train_auc",auc,on_epoch=True)

    def validation_step(self,batch, batch_idx):
        times, Y, mask, label = batch
        preds = self(times,Y,mask)
        if preds.shape[-1]==1:
            preds = preds[:,0]
            loss = self.loss_class(preds.double(),label)
        else:
            loss = self.loss_class(preds.double(),label.long())
        self.log("val_loss",loss,on_epoch=True)
        return {"Y":Y, "preds":preds, "T":times, "labels":label}
    
    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        if (self.hparams["data_type"]=="pMNIST") or (self.hparams["data_type"]=="Character"):
            preds = torch.nn.functional.softmax(preds,dim=-1).argmax(-1)
            accuracy = accuracy_score(labels.long().cpu().numpy(),preds.cpu().numpy())
            self.log("val_acc",accuracy,on_epoch=True)
        else:
            auc = roc_auc_score(labels.cpu().numpy(),preds.cpu().numpy())
            self.log("val_auc",auc,on_epoch=True)

        times = outputs[0]["T"]
        Y_sample = outputs[0]["Y"]

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = self.hparams.lr, weight_decay = self.hparams.weight_decay)

    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--Nc', type=int, default=32, help = "Dimension of the hidden vector")
        parser.add_argument('--delta_t', type=float, default=0.05, help = "integration step size")
        return parser