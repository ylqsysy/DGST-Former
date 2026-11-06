import torch
from torch import nn
import torch.nn.functional as F

class GraphOperators(nn.Module):
    def __init__(self, latent, hidden, adjacency, epsilon=0.01, graph_constraint=True, ph_constraints=True,compact_operators=False):
        super().__init__()
        i,j=torch.where(torch.triu(adjacency.detach().cpu()>0,diagonal=1))
        self.register_buffer("edge_i",i);self.register_buffer("edge_j",j)
        self.register_buffer("degree_scale",torch.bincount(torch.cat([i,j]),minlength=len(adjacency)).max().float().clamp_min(1))
        self.channel_j=nn.Parameter(torch.eye(latent)*0.1)
        self.channel_r=nn.Parameter(torch.eye(latent)*0.3)
        self.edge_gate=nn.Sequential(nn.Linear(2*(latent+hidden),hidden),nn.SiLU(),nn.Linear(hidden,1))
        self.gate=nn.Sequential(nn.Linear(latent+hidden,hidden),nn.SiLU(),nn.Linear(hidden,latent))
        self.global_p=nn.Linear(latent+hidden,4) if not (compact_operators and graph_constraint) else None
        self.global_q=nn.Linear(latent+hidden,4) if not (compact_operators and graph_constraint) else None
        self.epsilon=epsilon;self.graph_constraint=graph_constraint;self.ph_constraints=ph_constraints

    def coefficients(self,z,c,state_dependent=True):
        features=torch.cat([z,c],-1)
        gate=0.05+1.95*torch.sigmoid(self.gate(features))
        i,j=self.edge_i,self.edge_j
        if self.graph_constraint:
            edge=torch.tanh(self.edge_gate(torch.cat([features[:,i],features[:,j]],-1))).squeeze(-1)/self.degree_scale
            weight=(0.1+0.9*torch.sigmoid(edge*self.degree_scale))/self.degree_scale
            if not state_dependent:
                gate=torch.ones_like(gate);weight=torch.ones_like(weight)/self.degree_scale
            return gate,edge,weight,None,None
        p=torch.tanh(self.global_p(features))/z.shape[1]**0.5
        q=torch.tanh(self.global_q(features))/z.shape[1]**0.5
        if not state_dependent:
            gate=torch.ones_like(gate)
            p=torch.ones_like(p)/z.shape[1]**0.5
        return gate,None,None,p,q

    def interconnect(self,v,coeff):
        _,edge,_,p,q=coeff
        w=(self.channel_j+self.channel_j.T)*0.5 if self.ph_constraints else self.channel_j
        v=v@w
        if edge is not None:
            i,j=self.edge_i,self.edge_j
            out=torch.zeros_like(v).index_add(1,i,edge[...,None]*v[:,j])
            return out.index_add(1,j,(-edge if self.ph_constraints else edge.square()+0.1)[...,None]*v[:,i])
        first=p@torch.einsum("bnr,bnd->brd",q,v)
        second=q@torch.einsum("bnr,bnd->brd",p,v)
        return first-second if self.ph_constraints else first+second

    def dissipate(self,v,coeff):
        gate,_,weight,p,_=coeff
        root=gate.sqrt(); u=root*v
        w=self.channel_r@self.channel_r.T if self.ph_constraints else self.channel_r
        if weight is not None:
            i,j=self.edge_i,self.edge_j
            difference=(u[:,i]-u[:,j])@w*weight[...,None]
            out=torch.zeros_like(u).index_add(1,i,difference).index_add(1,j,-difference)
        else: out=p@torch.einsum("bnr,bnd->brd",p,u@w)
        return root*(out+self.epsilon*u)
