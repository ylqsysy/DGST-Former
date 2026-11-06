import torch
from torch import nn
from .embedding import TrafficInputEmbedding
from .transformer_encoder import TransformerContextEncoder
from .dissipative_dynamics import DissipativeDynamics
from .state_evolution import MultiStepEvolution
from .decoder import PredictionDecoder
from .gated_dynamics import GatedDiscreteDynamics,GatedMultiStepEvolution

class DGSTFormer(nn.Module):
    def __init__(self,config,adjacency):
        super().__init__();mc=config["model"];dc=config["data"]
        hidden=mc["hidden_dim"];latent=mc["latent_dim"];nodes=adjacency.shape[0]
        self.embedding=TrafficInputEmbedding(nodes,dc["input_length"],hidden)
        self.encoder=TransformerContextEncoder(hidden,mc["num_layers"],mc["num_heads"],mc["dropout"],mc.get('feed_forward_dim'),mc.get('activation','relu'))
        self.initializer=nn.Sequential(nn.Linear(hidden,latent),nn.ReLU() if mc.get('activation')=='relu' else nn.Tanh())
        self.future_query=nn.Parameter(torch.randn(dc["output_length"],hidden)*0.02)
        self.evolution_mode=mc.get("evolution_mode","rk4")
        if self.evolution_mode not in ["rk4","ode","direct","gated"]:raise ValueError("Unknown state evolution mode")
        if self.evolution_mode=='gated':self.dynamics=GatedDiscreteDynamics(latent,hidden,adjacency,**mc)
        elif self.evolution_mode!="direct":self.dynamics=DissipativeDynamics(latent,hidden,adjacency,**mc)
        else:
            self.dynamics=None
            width=mc.get("direct_width",148)
            self.direct_head=nn.Sequential(nn.Linear(hidden,width),nn.ReLU() if mc.get('activation')=='relu' else nn.SiLU(),nn.Linear(width,latent))
            i,j=torch.where(torch.triu(adjacency.detach().cpu()>0,diagonal=1))
            self.register_buffer("direct_edge_i",i);self.register_buffer("direct_edge_j",j)
        self.evolution=GatedMultiStepEvolution(mc.get('checkpoint_dynamics',True)) if self.evolution_mode=='gated' else MultiStepEvolution(mc.get('integration_steps',1),mc.get('dt',.1),mc.get("solver","rk4"),mc.get('checkpoint_dynamics',False))
        self.encoder_precision=mc.get('encoder_precision','float32')
        self.transformer_context=mc.get('transformer_context',True)
        if not self.transformer_context:
            if self.evolution_mode!='gated':raise ValueError('Context ablation requires gated evolution')
            self.encoder.temporal=None
            self.future_query.requires_grad_(False)
        self.contraction_rho=mc.get('contraction_rho',.99);self.jacobian_power_iterations=mc.get('jacobian_power_iterations',1)
        self.decoder=PredictionDecoder(latent)
        self.encoder_graph=mc.get("encoder_graph",mc["graph_constraint"])
        self.register_buffer("road_adjacency",adjacency.clone().float())
    @property
    def nfe(self): return self.dynamics.nfe if self.dynamics is not None else 0

    def encode_history(self,history_data,graph):
        if not self.transformer_context:
            embedded=self.embedding(history_data)
            context=embedded.mean(1)
            if graph is not None:
                i,j=graph;n=context.shape[1]
                degree=torch.bincount(torch.cat([i,j]),minlength=n).to(context).clamp_min(1)
                neighbors=torch.zeros_like(context).index_add(1,i,context[:,j]).index_add(1,j,context[:,i])
                context=context+self.encoder.graph_mix(neighbors/degree[None,:,None])
            return self.encoder.norm(context),embedded.permute(0,2,1,3)
        use_amp=self.encoder_precision=='bfloat16' and history_data.is_cuda and history_data.dtype==torch.float32
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=use_amp):
            context,temporal=self.encoder(self.embedding(history_data),graph)
        return context.to(history_data.dtype),temporal.to(history_data.dtype)

    def evolution_contexts(self,context):
        if not self.transformer_context:
            return context.new_zeros(context.shape[0],len(self.future_query),context.shape[1],context.shape[2])
        return context[:,None]+self.future_query[None,:,None,:]
    def forward(self,history_data,adjacency_matrix=None,return_diagnostics=False,closed=False,return_regularization=False):
        if history_data.shape[-1] != 1: raise ValueError("Only historical flow is supported")
        if adjacency_matrix is not None and (adjacency_matrix.shape != self.road_adjacency.shape or not torch.equal(adjacency_matrix,self.road_adjacency)):
            raise ValueError("Adjacency changed after construction; rebuild the model for a different road graph")
        if self.dynamics is not None:
            op=self.dynamics.operators;edges=(op.edge_i,op.edge_j)
        else:edges=(self.direct_edge_i,self.direct_edge_j)
        graph=edges if self.encoder_graph else None
        context,temporal=self.encode_history(history_data,graph)
        z=self.initializer(context)
        contexts=self.evolution_contexts(context)
        regularization=None
        if self.evolution_mode=="direct":states=z[:,None]+self.direct_head(contexts);trace=[]
        elif self.evolution_mode=='gated':
            states,trace,regularization=self.evolution(self.dynamics,z,contexts,return_diagnostics,closed,return_regularization,self.contraction_rho,self.jacobian_power_iterations)
        else:states,trace=self.evolution(self.dynamics,z,contexts,return_diagnostics,closed)
        prediction=self.decoder(states,history_data[:,-1])
        if return_diagnostics:
            return prediction,{"states":states,"initial_state":z,"trace":trace,"contexts":contexts,"temporal_context":temporal}
        if return_regularization:
            if regularization is None:raise ValueError('Jacobian regularization requires gated state evolution')
            return prediction,regularization
        return prediction
