"""
Created on Fri March 10 2017
@author: Wei Duan
"""
from torch.nn import Module
from torch import nn
import torch.nn.functional as F
class vgg(Module):
    # initialization
    def __init__(self):
        super(vgg, self).__init__()
        # parameters of Conv2d: nb_channels, nb_kernels, size_kernel
        self.conv1 = nn.Conv2d(3, 64, 3, padding = (1,1))
        self.conv2 = nn.Conv2d(64, 64, 3, padding = (1,1))
        self.pool = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(128, 128, 3, padding=(11))

        self.conv5 = nn.Conv2d(128, 256, 3, padding=(1,1))
        self.conv6 = nn.Conv2d(256, 256, 3, padding=(1,1))

        self.conv7 = nn.Conv2d(256, 512, 3, padding=(1,1))
        self.conv8 = nn.Conv2d(512, 512, 3, padding=(1, 1))

        self.fc1 = nn.Linear(512*3*3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.dropout = nn.Dropout(p = 0.5)
        self.softmax = nn.Softmax()

    # override forward function
    def forward(self, *x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv6(F.relu(self.conv5(x)))))))
        x = self.pool(F.relu(self.conv8(F.relu(self.conv8(F.relu(self.conv7(x)))))))
        x = self.pool(F.relu(self.conv8(F.relu(self.conv8(F.relu(self.conv7(x)))))))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))

