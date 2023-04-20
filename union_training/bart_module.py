import os
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
        scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        self._file = open(os.path.join(self.config.tmp_dir, 'changes_temp.txt'), 'w')

    def on_train_epoch_end(self):
        self._file.close()

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config.bart_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=buffer_collate
        )
        return loader

    def _write_changes(self, blk, key, value):
        self._file.write('{} {} {}\n'.format(blk.pos, key, value))

    def _intervention(self, bufs, labels, decoder_attention_masks, loss_bart):
        loss_bart = loss_bart.detach()
        with torch.no_grad():
            max_bs = self.config.bart_batch_size * 4
            for i in range(len(bufs)):
                temp_inputs = torch.zeros(2, 1024, dtype=torch.long, device=self.device)
                ids, attn_masks = bufs[i].export(out=temp_inputs, device=self.device)
                bs = len(bufs[i])
                ids = ids.view(1, -1).expand(bs, -1)
                attn_masks = attn_masks.view(1, -1).repeat(bs, 1)
                label = labels[i].view(1, -1).expand(bs, -1)
                decoder_attention_mask = decoder_attention_masks[i].view(1, -1).expand(bs, -1)
                blk_start, t = 0, 0
                for blk in bufs[i]:
                    blk_end = blk_start + len(blk)
                    attn_masks[t, blk_start: blk_end].zero_()
                    t += 1
                    blk_start = blk_end
                assert t == bs
                losses = []
                for j in range((bs - 1) // max_bs + 1):
                    l, r = max_bs * j, min(bs, max_bs * (j + 1))
                    result = self.summarizer(input_ids=ids[l:r], attention_mask=attn_masks[l:r], labels=label[l:r],
                                             decoder_attention_mask=decoder_attention_mask[l:r])
                    result = result.loss
                    losses.append(result)
                if len(losses) != 0:
                    losses_delta = torch.cat(losses, dim=0) - loss_bart[i]
                    # Label relevance
                    t = 0
                    for blk in bufs[i]:
                        if losses_delta[t] >= self.config.levelup_threshold and blk.relevance < 2:
                            self._write_changes(blk, 'relevance', blk.relevance + 1)
                        elif losses_delta[t] <= self.config.leveldown_threshold and blk.relevance > -2:
                            self._write_changes(blk, 'relevance', blk.relevance - 1)
                        t += 1
                    assert t == bs

    def training_step(self, bufs, batch_idx):
        inputs = torch.zeros(4, len(bufs), 1024, dtype=torch.long, device=self.device)
        for i, buf in enumerate(bufs):
            buf.export(out=(inputs[0, i], inputs[1, i]), device=self.device)
            inputs[2, i] = buf.summary['input_ids']
            inputs[3, i] = buf.summary['attention_mask']
        result = self.summarizer(input_ids=inputs[0], attention_mask=inputs[1], labels=inputs[2],
                                 decoder_attention_mask=inputs[3])
        loss_bart = result.loss
        self._intervention(bufs, inputs[2], inputs[3], loss_bart)
        loss_bart = loss_bart.mean()
        self.log('bart_loss', loss_bart, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.bart_batch_size)
        self.log('lr2', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.bart_batch_size)
        return loss_bart
