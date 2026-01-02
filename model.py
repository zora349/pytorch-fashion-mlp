import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self,x):
        logits = self.net(x)
        #print(logits.shape)
        return logits
