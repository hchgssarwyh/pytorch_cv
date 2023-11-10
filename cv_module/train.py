import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
import cv_module
import wandb

wandb_logger = WandbLogger(log_model="all")
dm = cv_module.data.CIFAR10DataModule(data_dir="dataset/", batch_size=cv_module.batch_size, num_workers=4)
trainer = L.Trainer(logger = wandb_logger, accelerator = "auto", devices = 1, min_epochs = 1, max_epochs = cv_module.epochs)
