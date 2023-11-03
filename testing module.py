import cv_module.data
import cv_module.neural_network
import cv_module.train
import cv_module

net = cv_module.neural_network.Net()
trainloader = cv_module.data.trainloader()

cv_module.train.train_nn(net, trainloader)

testloader = cv_module.data.testloader()

cv_module.neural_network.test_Net(net, testloader)

cv_module.neural_network.accuracy_for_classes(net, testloader)
