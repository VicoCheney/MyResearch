import logging
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utils import find_lastest_checkpoint
from bert_module import BertModule
from bart_module import BartModule
from data_driver import DataDriver
from metrics import evaluate


def train(config):
    os.makedirs(config.tmp_dir, exist_ok=True)

    data_driver = DataDriver(config)
    if not os.path.exists(os.path.join(config.tmp_dir, 'initialization.txt')):
        data_driver.SimCSE_initialize()
    data_driver.apply_initialization(config.tmp_dir)

    bert_logger = TensorBoardLogger(config.log_dir, name='bert_logger', version=config.version)
    bart_logger = TensorBoardLogger(config.log_dir, name='bart_logger', version=config.version)

    bert_model = BertModule(config)
    bart_model = BartModule(config)

    bert_checkpoint_path = os.path.join(config.save_dir, 'bert_model', f'version_{config.version}', 'checkpoints')
    bart_checkpoint_path = os.path.join(config.save_dir, 'bart_model', f'version_{config.version}', 'checkpoints')

    bert_min_epoch = find_lastest_checkpoint(bert_checkpoint_path, epoch=True) + 1

    logging.info(f'Continue training at epoch {bert_min_epoch}...')
    bert_checkpoint_callback = ModelCheckpoint(dirpath=bert_checkpoint_path, filename="{epoch}", save_top_k=-1)
    bart_checkpoint_callback = ModelCheckpoint(dirpath=bart_checkpoint_path, filename="{epoch}", save_top_k=-1)

    for epoch in range(bert_min_epoch, config.bert_epochs):
        bert_dataset = data_driver.build_random_buffer(num_samples=config.num_samples)
        bert_model.dataset = bert_dataset
        bert_trainer = Trainer(max_epochs=epoch + 1, precision=16, accelerator='gpu', devices=[0], logger=bert_logger, callbacks=[bert_checkpoint_callback])
        if epoch == 0:
            bert_trainer.fit(bert_model)
        else:
            bert_trainer.fit(bert_model, ckpt_path=os.path.join(bert_checkpoint_path, f'epoch={epoch - 1}.ckpt'))

    trained_bert = BertModule.load_from_checkpoint(find_lastest_checkpoint(
        os.path.join(config.save_dir, 'bert_model', f'version_{config.version}', 'checkpoints')), ).to("cuda:0").eval()

    bart_dataset = data_driver.build_compressed_dataset(trained_bert)
    bart_min_epoch = find_lastest_checkpoint(bart_checkpoint_path, epoch=True) + 1

    for epoch in range(bart_min_epoch, config.bart_epochs):
        bart_model.dataset = bart_dataset
        bart_trainer = Trainer(max_epochs=epoch + 1, precision=16, accelerator='gpu', devices=[0], logger=bart_logger, callbacks=[bart_checkpoint_callback])
        if epoch == 0:
            bart_trainer.fit(bart_model)
        else:
            bart_trainer.fit(bart_model, ckpt_path=os.path.join(bart_checkpoint_path, f'epoch={epoch - 1}.ckpt'))

        evaluate(config, mode='validation')