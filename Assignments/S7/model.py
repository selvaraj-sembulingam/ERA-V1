import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.025

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, bias=False),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=False),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=False),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 10, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(10))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.gap(x)
        x = x.view((x.shape[0],-1))
        x = F.log_softmax(x, dim=1)

        return x

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 11, kernel_size=3, stride=1, bias=False),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(11, 11, kernel_size=3, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(11, 14, kernel_size=3, stride=1, bias=False),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(14, 14, kernel_size=3, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(14, 16, kernel_size=3, stride=1, bias=False),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(10))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.gap(x)
        x = x.view((x.shape[0],-1))
        x = F.log_softmax(x, dim=1)

        return x

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 11, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(11),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(11, 11, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(11),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(11, 14, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(14, 14, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(14, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(10))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.gap(x)
        x = x.view((x.shape[0],-1))
        x = F.log_softmax(x, dim=1)

        return x

class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 11, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(11),
            nn.ReLU(),
            nn.Dropout(dropout_value))
        self.layer2 = nn.Sequential(
            nn.Conv2d(11, 11, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(11),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(11, 14, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Dropout(dropout_value))
        self.layer4 = nn.Sequential(
            nn.Conv2d(14, 14, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(14, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value))
        self.layer6 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.gap(x)
        x = x.view((x.shape[0],-1))
        x = F.log_softmax(x, dim=1)

        return x
