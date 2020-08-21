import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import time
import numpy as np
from pathlib import Path
from transformers import WEIGHTS_NAME, CONFIG_NAME
from pytorch_lightning.loggers import TensorBoardLogger


def init_seed():
    seed_val = 42
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_model(model, output_dir):

    output_dir = Path(output_dir)
    # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = output_dir / WEIGHTS_NAME
    output_config_file = output_dir / CONFIG_NAME

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    #src_tokenizer.save_vocabulary(output_dir)

def load_model():
    pass

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class TrainLightening(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super(TrainLightening, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, source, target):
        return self.model(source,target)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        source = batch[0].to(device)
        target = batch[1].to(device)
        loss, output = self.forward(source=source,target=target)
        self.logger.experiment.add_scalar("Loss/Train",
                                            torch.mean(loss),
                                            self.current_epoch)
        return {'loss': loss}

    # def training_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     logs = {'loss': avg_loss}
    #     self.logger.experiment.add_scalar("Loss/Train",
    #                                         avg_loss,
    #                                         self.current_epoch)

    #     return {'avg_train_loss': avg_loss, 'train_log': logs}


    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        source = batch[0].to(device)
        target = batch[1].to(device)
        loss, output = self.forward(source=source,target=target)
        labels = target.flatten()
        preds = torch.argmax(output, axis=2).flatten()
        self.logger.experiment.add_scalar("Loss/Val",
                                            torch.mean(loss),
                                            self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Val",
                                            torch.mean((preds==labels).float()),
                                            self.current_epoch)
        return {'val_loss': loss, 'val_accuracy': (preds==labels).float()}

    # def validation_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
    #     self.logger.experiment.add_scalar("Loss/Val",
    #                                         avg_loss,
    #                                         self.current_epoch)
    #     self.logger.experiment.add_scalar("Accuracy/Val",
    #                                         acc,
    #                                         self.current_epoch)
    #     logs = {'val_loss': avg_loss, 'val_acc': acc}
    #     return {'avg_val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        # REQUIRED
        return [self.optimizer], [self.scheduler]


def run_train(config, model, train_loader, eval_loader):
    init_seed()
    epochs = config.epochs
    output_path = './lightning'

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=config.lr)
    
    logger = TensorBoardLogger('tb_logs',name='off_note_model')
    model_ln = TrainLightening(model=model, optimizer=optimizer, scheduler=scheduler)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_epochs=epochs,
        weights_save_path=output_path,
        logger = logger,
        progress_bar_refresh_rate=1
        # use_amp=False,
    )
    trainer.fit(model_ln,train_loader, eval_loader)


if __name__ == '__main__':
    main()