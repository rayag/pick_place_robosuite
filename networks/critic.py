import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

import numpy as np

device = "cpu"

class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, goal_dim) -> None:
        super().__init__()
        self.input = nn.Linear(obs_dim + goal_dim + action_dim, 256).to(device)
        self.h1 = nn.Linear(256, 256).to(device)
        self.h2 = nn.Linear(256, 256).to(device)
        self.h3 = nn.Linear(256, 256).to(device)
        self.output = nn.Linear(256, 1).to(device)

    def forward(self, sg, a):
        x = F.relu(self.input(torch.cat([sg, a], 1)))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        out = self.output(x)
        return out

class CriticNetworkLowDim(nn.Module):
    def __init__(self, obs_dim, action_dim, goal_dim) -> None:
        super().__init__()
        self.low_dim = 3
        self.input = nn.Linear(obs_dim + goal_dim + self.low_dim, 256).to(device)
        self.h1 = nn.Linear(256, 256).to(device)
        self.h2 = nn.Linear(256, 256).to(device)
        self.h3 = nn.Linear(256, 256).to(device)
        self.output = nn.Linear(256, 1).to(device)

    def forward(self, sg, a):
        if len(a.size()) == 1:
            a = a[:self.low_dim]
        elif len(a.size()) == 2:
            a = a[:, :self.low_dim]
        x = F.relu(self.input(torch.cat([sg, a], 1)))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        out = self.output(x)
        return out