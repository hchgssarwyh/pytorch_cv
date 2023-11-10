import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv_module
import pytorch_lightning as L
from torch.utils.data import DataLoader
from torch.utils.data import random_split



class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))]
        )
    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train = True, download = True)
        datasets.CIFAR10(self.data_dir, train = False, download = True)
    def setup(self, stage):
        entire_dataset = datasets.CIFAR10(
            root = self.data_dir,
            train = True,
            transform = self.transform,
            download = False
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [45000, 5000])

        self.test_ds = datasets.CIFAR10(
            root = self.data_dir,
            train = False,
            transform = self.transform,
            download = False
        )
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True
            )
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size = 2*self.batch_size,
            num_workers = self.num_workers,
            shuffle = False
            )
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False
            )
    
    
import matplotlib.pyplot as plt
import numpy as np
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_random_images(loader):
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{cv_module.classes[labels[j]]:5s}' for j in range(batch_size)))
