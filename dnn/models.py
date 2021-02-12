from torch import nn
import torch.nn.functional as F


# Taken from repository here:
# https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/model.py
class LeNet(nn.Module):
    def __init__(self, n_classes=10):
        super(LeNet, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Taken from tutorial on Keras/TF flat model
# https://medium.com/analytics-vidhya/multi-layer-perceptron-using-keras-on-mnist-dataset-for-digit-classification-problem-relu-a276cbf05e97
class FlatNet(nn.Module):
    def __init__(self, n_classes=10):
        super(FlatNet, self).__init__()
        self.fc1 = nn.Linear(784, 364)
        self.fc2 = nn.Linear(364, 52)
        self.fc3 = nn.Linear(52, n_classes)
        self.drop_1 = nn.Dropout(0.5)
        self.drop_2 = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.drop_1(x)
        x = F.relu(self.fc2(x))
        x = self.drop_2(x)
        x = self.fc3(x)
        return x


# Basic logistic-regression model
class LR(nn.Module):
    def __init__(self, n_classes=10):
        super(LR, self).__init__()
        self.fc = nn.Linear(784, n_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.fc(x)
