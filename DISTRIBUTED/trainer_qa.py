import torch
import pytorch_lightning as pl

from torch.nn import functional as F
from transformers import Trainer

class QALightning(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, train_batch, batch_idx):
        outputs = self.model(**train_batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def forward(self, batch):
        return self.model(**batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, val_batch, batch_idx):
        outputs = self.model(**val_batch)
        loss = outputs.loss
        self.log("val_loss", loss, sync_dist=True)
        return loss
