import numpy as np 

import torch.nn as nn
import torch.nn.functional as F



DEPTH_MULT = 2

class ConvLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3):
        super(ConvLayer, self).__init__()
        self.ops = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.ops(x)


class FCLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCLayer, self).__init__()
        self.ops = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(inplace=True)
        )
        self.residual = input_size == output_size
    
    def forward(self, x):
        if self.residual:
            return (self.ops(x) + x) / np.sqrt(2)
        return self.ops(x)


def mixup(x, shuffle, lam, i, j):
    if shuffle is not None and lam is not None and i == j:
        x = lam * x + (1 - lam) * x[shuffle]
    return x


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        
        self.conv1 = ConvLayer(1, DEPTH_MULT * 32)
        self.conv2 = ConvLayer(DEPTH_MULT * 32, DEPTH_MULT * 32)
        self.conv3 = ConvLayer(DEPTH_MULT * 32, DEPTH_MULT * 32)
        self.conv4 = ConvLayer(DEPTH_MULT * 32, DEPTH_MULT * 32)
        
        self.conv5 = ConvLayer(DEPTH_MULT * 32, DEPTH_MULT * 64)
        self.conv6 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)
        self.conv7 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)
        self.conv8 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)
        self.conv9 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)
        self.conv10 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)
        
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = FCLayer(DEPTH_MULT * 64 * 7 * 7, DEPTH_MULT * 512)
        self.fc2 = FCLayer(DEPTH_MULT * 512, DEPTH_MULT * 512)
        self.fc3 = FCLayer(DEPTH_MULT * 512, DEPTH_MULT * 512)
        self.fc4 = FCLayer(DEPTH_MULT * 512, DEPTH_MULT * 512)
        self.projection = nn.Linear(DEPTH_MULT * 512, 10)
    
    def forward(self, x):
        if isinstance(x, list):
            x, shuffle, lam = x
        else:
            shuffle = None
            lam = None
        
        # Decide which layer to mixup
        j = np.random.randint(15)
        
        x = mixup(x, shuffle, lam, 0, j)
        x = self.conv1(x)
        x = mixup(x, shuffle, lam, 1, j)
        x = self.conv2(x)
        x = mixup(x, shuffle, lam, 2, j)
        x = self.conv3(x)
        x = mixup(x, shuffle, lam, 3, j)
        x = self.conv4(x)
        x = self.mp(x)
        
        x = mixup(x, shuffle, lam, 4, j)
        x = self.conv5(x)
        x = mixup(x, shuffle, lam, 5, j)
        x = self.conv6(x)
        x = mixup(x, shuffle, lam, 6, j)
        x = self.conv7(x)
        x = mixup(x, shuffle, lam, 7, j)
        x = self.conv8(x)
        x = mixup(x, shuffle, lam, 8, j)
        x = self.conv9(x)
        x = mixup(x, shuffle, lam, 9, j)
        x = self.conv10(x)
        x = self.mp(x)
        
        x = x.view(x.size(0), -1)
        x = mixup(x, shuffle, lam, 10, j)
        x = self.fc1(x)
        x = mixup(x, shuffle, lam, 11, j)
        x = self.fc2(x)
        x = mixup(x, shuffle, lam, 12, j)
        x = self.fc3(x)
        x = mixup(x, shuffle, lam, 13, j)
        x = self.fc4(x)
        x = mixup(x, shuffle, lam, 14, j)
        x = self.projection(x)
        
        return x
