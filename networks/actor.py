import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

import numpy as np

device = "cpu"

class ActorNetwork(nn.Module): 
    def __init__(self, obs_dim, action_dim, action_high = 1.0, action_low = 0.0) -> None:
        super(ActorNetwork, self).__init__()
        self.action_high = torch.FloatTensor(action_high).to(device)
        self.action_low = torch.FloatTensor(action_low).to(device)
        self.input = nn.Linear(obs_dim, 256).to(device) #TODO: allow custom layer sizes
        self.h1 = nn.Linear(256, 256).to(device)
        self.h2 = nn.Linear(256, 256).to(device)
        self.h3 = nn.Linear(256, 256).to(device)
        self.output = nn.Linear(256, action_dim).to(device)

    def forward(self, obs):
        x = F.relu(self.input(obs))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        action = torch.tanh(self.output(x)) * self.action_high
        return action

class ActorNetworkLowDim(nn.Module): 
    def __init__(self, obs_dim, action_dim, action_high = 1.0, action_low = 0.0) -> None:
        super(ActorNetworkLowDim, self).__init__()
        self.action_dim = action_dim
        self.action_low_dim = 3
        self.action_high = torch.FloatTensor(np.full(shape=(self.action_low_dim), fill_value=1)).to(device)
        self.action_low = torch.FloatTensor(np.full(shape=(self.action_low_dim), fill_value=-1)).to(device)
        self.input = nn.Linear(obs_dim, 256).to(device) #TODO: allow custom layer sizes
        self.h1 = nn.Linear(256, 256).to(device)
        self.h2 = nn.Linear(256, 256).to(device)
        self.h3 = nn.Linear(256, 256).to(device)
        self.output = nn.Linear(256, self.action_low_dim).to(device)

    def forward(self, obs):
        x = F.relu(self.input(obs))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        action = torch.tanh(self.output(x)) * self.action_high
        if len(action.size())==1:
            action_full = torch.zeros(self.action_dim)
            action_full[:3] = action[:3]
            action_full[-1] = action[-1]
        if len(action.size()) == 2:
            action_full = torch.zeros(action.size()[0], self.action_dim)
            action_full[:,:3] = action[:,:3]
            action_full[:,-1] = action[:,-1]
        return action_full