import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import BertPreTrainedModel, RobertaConfig, RobertaModel
from buffer import buffer_collate


class BertModule(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.dataset = None
        self.config = config
        self.compresser = Compresser.from_pretrained(config.compresser_model)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.compresser.parameters(), self.config.lr1)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config.bert_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=buffer_collate
        )
        return loader

    def training_step(self, bufs, batch_idx):
        inputs = torch.zeros(4, len(bufs), 512, dtype=torch.long, device=self.device)
        for i, buf in enumerate(bufs):
            buf.export(out=(inputs[0, i], inputs[1, i]), device=self.device)
        for i, buf in enumerate(bufs):
            buf.export_relevance(out=inputs[3, i])
        loss_bert, logits = self.compresser(*inputs[:3], labels=inputs[3])
        self.log('bert_loss', loss_bert, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.bert_batch_size)
        self.log('lr1', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=self.config.bert_batch_size)
        return loss_bert


class Compresser(BertPreTrainedModel):

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(Compresser, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = logits
        if labels is not None:
            labels = labels.type_as(logits)
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss, logits)

        return outputs