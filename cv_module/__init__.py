import wandb
import torch

batch_size = 64
epochs = 15
learning_rate = 2e-4


classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

random_seed = 42
torch.manual_seed(random_seed);
