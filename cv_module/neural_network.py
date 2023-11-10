from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import cv_module
import pytorch_lightning as L
import torchmetrics

 
class Net(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128,kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)


        self.fc1 = nn.Linear(256*4*4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task = 'multiclass', num_classes = num_classes)
        self.f1_score = torchmetrics.F1Score(task = 'multiclass', num_classes = num_classes)

    def forward(self, x):
        x = F.relu(self.dropout(self.conv1(x)))
        x = self.pool(F.relu(self.dropout(self.conv2(x))))

        x = F.relu(self.dropout(self.conv3(x)))
        x = self.pool(F.relu(self.dropout(self.conv4(x))))

        x = F.relu(self.dropout(self.conv5(x)))
        x = self.pool(F.relu(self.dropout(self.conv6(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = F.log_softmax(self.dropout(self.fc3(x)))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = self.loss_fn(scores, y)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'train_loss':loss, 'train_accuracy':accuracy, 'train_f1_score':f1_score})
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = self.loss_fn(scores, y)
        self.log('val_loss',loss)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = self.loss_fn(scores, y)
        accuracy = self.accuracy(scores, y)
        self.log('test_accuracy', accuracy)
        return loss
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = cv_module.learning_rate) 
        
        
def save_Net(net, PATH):
    torch.save(net.state_dict(), PATH)



def accuracy_for_classes(net, testloader):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in cv_module.classes}
    total_pred = {classname: 0 for classname in cv_module.classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[cv_module.classes[label]] += 1
                total_pred[cv_module.classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
