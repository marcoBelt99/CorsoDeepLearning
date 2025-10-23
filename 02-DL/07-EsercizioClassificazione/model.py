import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import parameters as param # Parametri


class MultiLabelNN(nn.Module):
    def __init__(self, num_classi):
        super(MultiLabelNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * (param.image_size // 8) * (param.image_size // 8), 128)
        self.fc2 = nn.Linear(128, num_classi)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * (param.image_size // 8) * (param.image_size // 8)) # Flatten
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)) # Funzione sigmoide per classificazione (DEVO USARE LA SOFTMAX, ma qui non so perchè usa la sigmoide)
        return x
    
