import torch
from torch import nn

class TransformerContextEncoder(nn.Module):
    def __init__(self, hidden, layers, heads, dropout,feed_forward_dim=None,activation='relu'):
        super().__init__()
        block=nn.TransformerEncoderLayer(hidden,heads,feed_forward_dim or hidden*2,dropout,activation=activation,batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.graph_mix=nn.Linear(hidden,hidden)
        self.norm=nn.LayerNorm(hidden)
        self.temporal_chunk_size=32768
    def forward(self,x,graph):
        b,t,n,d=x.shape
        flat=x.permute(0,2,1,3).reshape(b*n,t,d)
        encoded=self.temporal(flat) if len(flat)<=self.temporal_chunk_size else torch.cat([self.temporal(part) for part in flat.split(self.temporal_chunk_size)],0)
        temporal=encoded.reshape(b,n,t,d)
        context=temporal[:,:,-1]
        if graph is not None:
            i,j=graph
            degree=torch.bincount(torch.cat([i,j]),minlength=n).to(x).clamp_min(1)
            neighbors=torch.zeros_like(context)
            neighbors=neighbors.index_add(1,i,context[:,j]).index_add(1,j,context[:,i])
            context=context+self.graph_mix(neighbors/degree[None,:,None])
        return self.norm(context), temporal
