import torch
from torch import nn
from torch.nn import functional as F


class RoadAggregation(nn.Module):
    def __init__(self,adjacency):
        super().__init__();i,j=torch.where(torch.triu(adjacency.detach().cpu()>0,diagonal=1))
        self.register_buffer('edge_i',i);self.register_buffer('edge_j',j)
        self.register_buffer('degree',torch.bincount(torch.cat([i,j]),minlength=len(adjacency)).float())
    def forward(self,z):
        i,j=self.edge_i,self.edge_j
        neighbors=torch.zeros_like(z).index_add(1,i,z[:,j]).index_add(1,j,z[:,i])
        return (neighbors+z)/(self.degree+1)[None,:,None]
    def transpose(self,z):
        scaled=z/(self.degree+1)[None,:,None]
        i,j=self.edge_i,self.edge_j
        return scaled+torch.zeros_like(z).index_add(1,i,scaled[:,j]).index_add(1,j,scaled[:,i])


class GatedDiscreteDynamics(nn.Module):
    def __init__(self,latent,hidden,adjacency,**options):
        super().__init__();self.operators=RoadAggregation(adjacency)
        self.gamma_min=options.get('gamma_min',.01);self.gamma_max=options.get('gamma_max',.10)
        if not 0<=self.gamma_min<=self.gamma_max<1:raise ValueError('Require 0 <= gamma_min <= gamma_max < 1')
        self.gamma_mode=options.get('gamma_mode','learned')
        if self.gamma_mode not in ['learned','fixed','none']:raise ValueError('Unknown gamma mode')
        self.gamma_fixed=options.get('gamma_fixed',(self.gamma_min+self.gamma_max)/2)
        self.gamma_conditioning=options.get('gamma_conditioning',True)
        self.gamma_state_dependent=options.get('gamma_state_dependent',True)
        self.update_conditioning=options.get('update_conditioning',True)
        self.graph_constraint=options.get('graph_constraint',True)
        self.delta_scale=options.get('delta_scale',1.)
        self.gate=nn.Sequential(nn.Linear(latent+hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.update=nn.Sequential(nn.Linear(2*latent+hidden,hidden),nn.ReLU(),nn.Linear(hidden,latent))
        if self.gamma_mode!='learned':
            for parameter in self.gate.parameters():parameter.requires_grad_(False)
        self.nfe=0;self._power_generators={};self.power_seed=options.get('jacobian_seed',42)

    def gamma(self,z,c):
        if self.gamma_mode=='none':return torch.zeros_like(z[...,:1])
        if self.gamma_mode=='fixed':return torch.full_like(z[...,:1],self.gamma_fixed)
        context=c if self.gamma_conditioning else torch.zeros_like(c)
        state=z if self.gamma_state_dependent else torch.zeros_like(z)
        return self.gamma_min+(self.gamma_max-self.gamma_min)*torch.sigmoid(self.gate(torch.cat([state,context],-1)))

    def transition(self,z,c):
        self.nfe+=1
        neighbor=self.operators(z) if self.graph_constraint else torch.zeros_like(z)
        context=c if self.update_conditioning else torch.zeros_like(c)
        delta=self.delta_scale*self.update(torch.cat([z,neighbor,context],-1))
        return (1-self.gamma(z,c))*z+delta

    @staticmethod
    def storage(z):return .5*z.square().sum((-1,-2))

    def linearization(self,z,c):
        latent=z.shape[-1]
        neighbor=self.operators(z) if self.graph_constraint else torch.zeros_like(z)
        context=c if self.update_conditioning else torch.zeros_like(c)
        mask=(self.update[0](torch.cat([z,neighbor,context],-1))>0).to(z.dtype)
        gamma=self.gamma(z,c)
        if self.gamma_mode=='learned' and self.gamma_state_dependent:
            gate_context=c if self.gamma_conditioning else torch.zeros_like(c)
            gate_pre=self.gate[0](torch.cat([z,gate_context],-1))
            sigmoid=torch.sigmoid(self.gate[2](F.relu(gate_pre)))
            gate_grad=F.linear((gate_pre>0).to(z.dtype)*self.gate[2].weight,
                              self.gate[0].weight[:,:latent].T)
            gate_grad=gate_grad*((self.gamma_max-self.gamma_min)*sigmoid*(1-sigmoid))
        else:gate_grad=torch.zeros_like(z)
        return gamma,gate_grad,mask

    def state_jvp(self,z,v,factors):
        gamma,gate_grad,mask=factors;latent=z.shape[-1];weight=self.update[0].weight
        hidden=F.linear(v,weight[:,:latent])
        if self.graph_constraint:hidden=hidden+F.linear(self.operators(v),weight[:,latent:2*latent])
        return (1-gamma)*v-z*(gate_grad*v).sum(-1,keepdim=True)+self.delta_scale*F.linear(mask*hidden,self.update[2].weight)

    def state_vjp(self,z,u,factors):
        gamma,gate_grad,mask=factors;latent=z.shape[-1];weight=self.update[0].weight
        hidden=mask*F.linear(u,self.update[2].weight.T)
        result=(1-gamma)*u-gate_grad*(z*u).sum(-1,keepdim=True)+self.delta_scale*F.linear(hidden,weight[:,:latent].T)
        if self.graph_constraint:result=result+self.delta_scale*self.operators.transpose(F.linear(hidden,weight[:,latent:2*latent].T))
        return result

    def spectral_estimate(self,z,c,iterations=1,create_graph=True):
        if iterations<1:raise ValueError('At least one power iteration is required')
        with torch.enable_grad():
            factors=self.linearization(z,c)
            key=str(z.device)
            if key not in self._power_generators:self._power_generators[key]=torch.Generator(device=z.device).manual_seed(self.power_seed)
            v=torch.randn(z.shape,device=z.device,dtype=z.dtype,generator=self._power_generators[key])
            normalize=lambda t:t/t.flatten(1).norm(dim=1).clamp_min(1e-12).reshape(-1,1,1)
            with torch.no_grad():
                detached=tuple(t.detach() for t in factors);state=z.detach();v=normalize(v)
                for _ in range(iterations):
                    u=normalize(self.state_jvp(state,v,detached))
                    v=normalize(self.state_vjp(state,u,detached))
            sigma=self.state_jvp(z,v.detach(),factors).flatten(1).norm(dim=1)
        return sigma if create_graph else sigma.detach()

    def spectral_estimate_autodiff(self,z,c,iterations=1,create_graph=True):
        if iterations<1:raise ValueError('At least one power iteration is required')
        with torch.enable_grad():
            state=z if z.requires_grad else z.detach().requires_grad_(True)
            key=str(state.device)
            if key not in self._power_generators:self._power_generators[key]=torch.Generator(device=state.device).manual_seed(self.power_seed)
            v=torch.randn(state.shape,device=state.device,dtype=state.dtype,generator=self._power_generators[key])
            normalize=lambda t:t/t.flatten(1).norm(dim=1).clamp_min(1e-12).reshape(-1,1,1)
            v=normalize(v).detach()
            function=lambda value:self.transition(value,c)
            for _ in range(iterations):
                primal,jv=torch.func.jvp(function,(state,),(v,))
                u=normalize(jv).detach()
                vjp=torch.autograd.grad(primal,state,u,retain_graph=True,create_graph=False)[0]
                v=normalize(vjp).detach()
            _,jv=torch.func.jvp(function,(state,),(v,))
            sigma=jv.flatten(1).norm(dim=1)
        return sigma if create_graph else sigma.detach()


class GatedMultiStepEvolution(nn.Module):
    def __init__(self,checkpoint_dynamics=True):
        super().__init__();self.checkpoint_dynamics=checkpoint_dynamics
    def forward(self,dynamics,z,contexts,diagnostics=False,closed=False,regularize=False,rho=.99,iterations=1):
        if closed:raise ValueError('Gated evolution has no closed-system mode')
        states=[];trace=[];penalties=[];sigmas=[];dynamics.nfe=0
        for c in contexts.unbind(1):
            if regularize:
                sigma=dynamics.spectral_estimate(z,c,iterations,True)
                sigmas.append(sigma);penalties.append(torch.relu(sigma-rho).square())
            if diagnostics:trace.append({'gamma':dynamics.gamma(z,c)})
            if self.checkpoint_dynamics and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                z=checkpoint(dynamics.transition,z,c,use_reentrant=False,preserve_rng_state=False)
            else:z=dynamics.transition(z,c)
            states.append(z)
        info={'dissipation_loss':torch.stack(penalties,1).mean() if penalties else z.new_zeros(()),
              'jacobian_norm_estimates':torch.stack(sigmas,1) if sigmas else None}
        return torch.stack(states,1),trace,info
