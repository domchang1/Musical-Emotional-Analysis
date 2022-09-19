import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (1))
        self.conv2 = nn.Conv2d(1,1,1)
        

    def forward(self, x):
        conv = self.conv2(F.relu(self.conv1(x)))
        return torch.flatten(conv)
