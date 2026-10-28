import torch
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup

class QALightning(pl.LightningModule):
    def __init__(self, model, learning_rate, warmup_steps):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

    def training_step(self, train_batch, batch_idx):
        outputs = self.model(**train_batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def forward(self, batch):
        return self.model(**batch)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() 
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() 
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step",
                "frequency": 1,
            }
        }

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
        

