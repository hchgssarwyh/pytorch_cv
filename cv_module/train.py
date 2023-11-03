import torch.optim as optim
import torch.nn as nn
import wandb
import cv_module.neural_network

run = wandb.init(
    project="pytorch_classification",
    notes="My first experiment",
)

wandb.config = {"epochs": 4, "learning_rate": 0.0005, "batch_size": 4}
def train_nn(net, trainset):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)

    for epoch in range(4):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainset, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                wandb.log({'loss': running_loss/2000})
                running_loss = 0.0
    cv_module.neural_network.save_Net(net, 'model')            
    print('Finished Training')
