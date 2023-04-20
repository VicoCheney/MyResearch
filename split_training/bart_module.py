import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import BartForConditionalGeneration
from buffer import buffer_collate


class BartModule(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.dataset = None
        self.config = config
        self.summarizer = BartForConditionalGeneration.from_pretrained(config.summarizer_model)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.summarizer.parameters(), self.config.lr2)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config.bart_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=buffer_collate
        )
        return loader

    def training_step(self, bufs, batch_idx):
        inputs = torch.zeros(4, len(bufs), 1024, dtype=torch.long, device=self.device)
        for i, buf in enumerate(bufs):
            buf.export(out=(inputs[0, i], inputs[1, i]), device=self.device)
            inputs[2, i] = buf.summary['input_ids']
            inputs[3, i] = buf.summary['attention_mask']
        result = self.summarizer(input_ids=inputs[0], attention_mask=inputs[1], labels=inputs[2], decoder_attention_mask=inputs[3])
        loss_bart = result.loss.mean()
        self.log('bart_loss', loss_bart, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.bart_batch_size)
        self.log('lr2', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.bart_batch_size)
        return loss_bart
