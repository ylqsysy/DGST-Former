import torch
from torch import nn
import torch.nn.functional as F
from .graph_interaction import GraphOperators

class StorageFunction(nn.Module):
    def __init__(self,dim):
        super().__init__();self.raw_mass=nn.Parameter(torch.zeros(dim));self.raw_nonlinear=nn.Parameter(torch.full((dim,),-2.0))
    def forward(self,z):
        logcosh=F.softplus(2*z)-z-torch.log(z.new_tensor(2.0))
        return (0.5*(F.softplus(self.raw_mass)+0.1)*z.square()+F.softplus(self.raw_nonlinear)*logcosh).sum((-1,-2))
    def gradient(self,z):
        return (F.softplus(self.raw_mass)+0.1)*z+F.softplus(self.raw_nonlinear)*torch.tanh(z)

class DissipativeDynamics(nn.Module):
    def __init__(self,latent,hidden,adjacency,**options):
        super().__init__()
        self.storage=StorageFunction(latent)
        self.operators=GraphOperators(latent,hidden,adjacency,options.get("epsilon",0.01),options.get("graph_constraint",True),options.get("ph_constraints",True),options.get("compact_operators",False))
        self.input_map=nn.Linear(hidden,latent)
        self.input_gate=nn.Linear(latent+hidden,latent)
        self.conditioning=options.get("conditioning",True)
        self.operator_conditioning=options.get("operator_conditioning",self.conditioning)
        self.external_input=options.get("external_input",self.conditioning)
        self.state_dependent=options.get("state_dependent",True)
        self.has_dissipation=options.get("dissipation",True)
        self.scale=options.get("dissipation_scale",1.0)
        if options.get("epsilon",0.01)<0 or self.scale<0:raise ValueError("epsilon and dissipation_scale must be nonnegative")
        self.nfe=0
    def _terms(self,z,c,closed=False):
        operator_context=c if self.operator_conditioning else torch.zeros_like(c)
        coeff=self.operators.coefficients(z,operator_context,self.state_dependent)
        grad=self.storage.gradient(z)
        j=self.operators.interconnect(grad,coeff)
        r=self.operators.dissipate(grad,coeff)*self.scale if self.has_dissipation else torch.zeros_like(z)
        port_input=torch.tanh(self.input_map(c))
        coupling=torch.sigmoid(self.input_gate(torch.cat([z,c],-1)))
        forcing=coupling*port_input
        if closed or not self.external_input:
            forcing=torch.zeros_like(z);port_input=torch.zeros_like(port_input)
        rhs=j-r+forcing
        return rhs,grad,r,forcing,coupling,port_input,coeff
    def components(self,z,c,closed=False):
        rhs,grad,r,forcing,coupling,port_input,coeff=self._terms(z,c,closed)
        port_output=coupling*grad
        supply=(grad*forcing).sum((-1,-2));loss=(grad*r).sum((-1,-2))
        return rhs,{"supply":supply,"loss":loss,"balance_residual":(grad*rhs).sum((-1,-2))-supply+loss,"gate":coeff[0],"port_input":port_input,"port_output":port_output}
    def forward(self,z,c,closed=False):
        self.nfe+=1
        return self._terms(z,c,closed)[0]
