import torch
from torch import nn

def rk4_step(function,z,c,dt,closed=False):
    k1=function(z,c,closed);k2=function(z+dt*k1/2,c,closed)
    k3=function(z+dt*k2/2,c,closed);k4=function(z+dt*k3,c,closed)
    return z+dt*(k1+2*k2+2*k3+k4)/6

def euler_step(function,z,c,dt,closed=False):return z+dt*function(z,c,closed)

def heun_step(function,z,c,dt,closed=False):
    first=function(z,c,closed);second=function(z+dt*first,c,closed)
    return z+dt*(first+second)/2

class MultiStepEvolution(nn.Module):
    def __init__(self,steps=1,dt=0.1,solver="rk4",checkpoint_dynamics=False):
        super().__init__()
        if steps < 1 or dt <= 0: raise ValueError("Integration steps and dt must be positive")
        if solver not in ["rk4","heun","euler"]:raise ValueError("Unknown fixed-step solver")
        self.steps=steps;self.dt=dt;self.solver=solver;self.checkpoint_dynamics=checkpoint_dynamics
    def forward(self,dynamics,z,contexts,diagnostics=False,closed=False):
        states=[];trace=[];dynamics.nfe=0
        for c in contexts.unbind(1):
            def advance(state,context):
                for _ in range(self.steps):
                    state={"rk4":rk4_step,"heun":heun_step,"euler":euler_step}[self.solver](dynamics,state,context,self.dt/self.steps,closed)
                return state
            if self.checkpoint_dynamics and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                z=checkpoint(advance,z,c,use_reentrant=False,preserve_rng_state=False)
            else:z=advance(z,c)
            states.append(z)
            if diagnostics:
                _,info=dynamics.components(z,c,closed)
                trace.append({"storage":dynamics.storage(z),**info})
        return torch.stack(states,1),trace
