from argparse import Namespace
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
from modules.graph import GraphConvolution
from modules.inner_product import InnerProductDecoder
from modules.linear import LinearBlock

class GraphVAE(nn.Module):   
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, hidden_decoder, dropout, meanMin=1e-5,meanMax=1e6,thetaMin=1e-5,thetaMax=1e6) -> None:
        super(GraphVAE, self).__init__()
        self.gc = GraphConvolution(input_dim=input_feat_dim, output_dim=hidden_dim1, dropout=dropout, act=F.leaky_relu) 
        self.gc_mu = GraphConvolution(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=F.leaky_relu) 
        self.gc_logvar = GraphConvolution(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=F.leaky_relu) 
            
        self.adjacency = InnerProductDecoder(dropout=dropout, act=lambda x: x) ## Reconstruct Adj
    
        self.latent = LinearBlock(
            input_dim = hidden_dim2, 
            output_dim = hidden_decoder, 
            dropout = dropout, 
            act = F.leaky_relu, 
            batchnorm = True)
        
        self.pi = LinearBlock(
            input_dim=hidden_decoder, 
            output_dim=input_feat_dim, 
            dropout=0, 
            act = torch.sigmoid, 
            batchnorm = False,
            bias = True) 
        
        self.theta = LinearBlock(
            input_dim = hidden_decoder, 
            output_dim = input_feat_dim, 
            dropout = 0, 
            act = lambda x: torch.clamp(input=F.softplus(x), min=thetaMin, max=thetaMax),
            batchnorm = False,
            bias = True)

        self.mean = LinearBlock(
            input_dim = hidden_decoder, 
            output_dim = input_feat_dim, 
            dropout = 0, 
            act = lambda x: torch.clamp(input=torch.exp(input=x), min=meanMin, max=meanMax),
            batchnorm = False,
            bias = True)

    def encode(self, x, adj) -> Tuple[FloatTensor, FloatTensor]:
        hidden1 = self.gc(x, adj)
        return self.gc_mu(hidden1, adj), self.gc_logvar(hidden1, adj)

    def reparameterize(self, mu, logvar) -> FloatTensor:
        std = torch.exp(input=logvar)
        eps = torch.randn_like(input=std)
        return eps.mul(other=std).add_(other=mu)
    
        # if self.training:
        #     std = torch.exp(logvar)
        #     eps = torch.randn_like(std)
        #     return eps.mul(std).add_(mu)
        # else:
        #     return mu

    def decode(self, z: FloatTensor) -> Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor]:
        output = self.latent(z)
        pi = self.pi(output)
        theta = self.theta(output)
        mean = self.mean(output)
        return output, pi, theta, mean
    
    def forward(self, x: FloatTensor, adj: LongTensor) -> Namespace:
        output = Namespace()
        output.mu, output.logvar = self.encode(x=x, adj=adj)
        output.z = self.reparameterize(mu=output.mu, logvar=output.logvar)
        output.adj_recon = self.adjacency(output.z)
        output.features_recon = self.decode(z=output.z)
        return output