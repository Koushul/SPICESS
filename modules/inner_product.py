import torch.nn.functional as F
import torch.nn as  nn
import torch

class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self, dropout, act=lambda x: x, latent_dim=None):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
        self.latent_dim = latent_dim

    def forward(self, inputs):
        inputs=F.dropout(inputs, self.dropout, training=self.training)
        x=torch.mm(inputs, inputs.t())
        outputs = self.act(x)
        return outputs
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + ' (' + str(self.latent_dim) + ' -> ' + 'NxN' + ')'
       
