import torch
import pytorch_lightning as pl

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
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
        return optimizer

    def validation_step(self, val_batch, batch_idx):
        outputs = self.model(**val_batch)

        if outputs.loss is not None:
            loss = outputs.loss
            self.log("val_loss", loss, sync_dist=True)

        labels_start = val_batch.get("start_positions")
        labels_end = val_batch.get("end_positions")

        if labels_start is not None and labels_end is not None:
            preds_start = torch.argmax(outputs.start_logits, dim=1)
            preds_end = torch.argmax(outputs.end_logits, dim=1)
            
            exact_match = ((preds_start == val_batch["start_positions"]) & 
                        (preds_end == val_batch["end_positions"])).float().mean()
            
            self.log("val_em", exact_match, sync_dist=True)

        return outputs.loss
        

