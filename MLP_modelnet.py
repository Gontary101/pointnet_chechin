import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

#we need to add batch normalization


class MLP(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1) #flatten the input
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1) #activation for the classes 
        return x


