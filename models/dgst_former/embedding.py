import torch
from torch import nn

class TrafficInputEmbedding(nn.Module):
    def __init__(self, nodes, length, hidden):
        super().__init__()
        self.value=nn.Linear(1,hidden)
        self.node=nn.Parameter(torch.randn(1,1,nodes,hidden)*0.02)
        self.time=nn.Parameter(torch.randn(1,length,1,hidden)*0.02)
    def forward(self,x): return self.value(x)+self.node+self.time
