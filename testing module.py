import cv_module.data
import cv_module.neural_network
import cv_module.train
import cv_module
import pytorch_lightning as L
import wandb
from cv_module.train import trainer, dm

wandb.init(
    project="pytorch_classification",

    config={
    "learning_rate": cv_module.learning_rate,
    "dataset": "CIFAR-10",
    "batch_size": cv_module.batch_size,
    "epochs": cv_module.epochs,
    }
)
net = cv_module.neural_network.Net(len(cv_module.classes))

trainer.fit(net,dm)

trainer.validate(net, dm)

trainer.test(net, dm)

cv_module.neural_network.accuracy_for_classes(net, dm.test_dataloader())

wandb.finish()

cv_module.neural_network.saveNet(net,'')
