from torch import nn

class PredictionDecoder(nn.Module):
    def __init__(self,latent):
        super().__init__();self.network=nn.Linear(latent,1)
    def forward(self,states,last_observation):
        return self.network(states)+last_observation[:,None]
