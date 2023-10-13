## Author: Koushul Ramjattun
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .modules.graph import GraphConvolution
from .modules.inner_product import InnerProductDecoder
from .modules.infomax import ContrastiveGraph

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphEncoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.prelu2 = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        return x
    
def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

class InfoMaxVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        dropout=0,
        latent_dim=32,
        encoder_dim=128,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder_dim = encoder_dim
        self.dropout = dropout
                
        self.encoders = nn.ModuleList([
            ContrastiveGraph(
                hidden_channels=encoder_dim, 
                encoder=GraphEncoder(input_dim[0], encoder_dim),
                summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
                corruption=corruption),
            ContrastiveGraph(
                hidden_channels=encoder_dim, 
                encoder=GraphEncoder(input_dim[1], encoder_dim),
                summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
                corruption=corruption),
        ])
        
        
        dim1, dim2 = input_dim
        
        # self.encoders = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(dim1, encoder_dim*2),
        #         nn.BatchNorm1d(encoder_dim*2),
        #         nn.LeakyReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(encoder_dim*2, encoder_dim),
        #         nn.BatchNorm1d(encoder_dim),
        #         nn.LeakyReLU(),
        #     ),
        #     nn.Sequential(
        #         nn.Linear(dim2, encoder_dim*2),
        #         nn.BatchNorm1d(encoder_dim*2),
        #         nn.LeakyReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(encoder_dim*2, encoder_dim),
        #         nn.BatchNorm1d(encoder_dim),
        #         nn.LeakyReLU(),
        #     ),
        # ])
        
        self.fc_mus = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                nn.Linear(encoder_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.LeakyReLU(),
            )
        ])
    
        self.fc_vars = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                nn.Linear(encoder_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.LeakyReLU(),
            )
        ])
        
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, dim1),
                nn.BatchNorm1d(dim1),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim1, 2*dim1),
                nn.BatchNorm1d(2*dim1),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(2*dim1, dim1),
            ), 
            nn.Sequential(
                nn.Linear(latent_dim, dim2),
                nn.BatchNorm1d(dim2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim2, 2*dim2),
                nn.BatchNorm1d(2*dim2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(2*dim2, dim2),
            ), 
        ])
        
        self.adjacency = InnerProductDecoder(
            dropout=dropout, 
            act=lambda x: x)

    def encode(self, X, A):
        edge_index = A.nonzero().t().contiguous()
        gex_pos_z, gex_neg_z, gex_summary = self.encoders[0](X[0], edge_index)
        pex_pos_z, pex_neg_z, pex_summary = self.encoders[1](X[1], edge_index)                
        return [gex_pos_z, gex_neg_z, gex_summary, pex_pos_z, pex_neg_z, pex_summary]
    
    # def encode(self, X, A):
    #     gex_neg_z = gex_pos_z = self.encoders[0](X[0])
    #     pex_neg_z = pex_pos_z = self.encoders[1](X[1])                
    #     gex_summary = pex_summary = None
    #     return [gex_pos_z, gex_neg_z, gex_summary, pex_pos_z, pex_neg_z, pex_summary]
        
    def refactor(self, X, A):
        index = range(2)
        zs = []
        mus = []
        logvars = []
        for x, i in zip(X, index):
            mu = self.fc_mus[i](x)
            logvar = self.fc_vars[i](x)
            std = torch.exp(logvar / 2)
            if not self.training:
                zs.append(mu)
            else:
                std = std + 1e-7
                q = torch.distributions.Normal(mu, std)
                zs.append(q.rsample())
            mus.append(mu)
            logvars.append(logvar)
        return zs, mus, logvars

    def combine(self, Z):
        """This moves correspondent latent embeddings 
        in similar directions over the course of training 
        and is key in the formation of similar latent spaces."""         
        mZ = 0.5*(Z[0] + Z[1])        
        return [mZ, mZ]

    def decode(self, X):
        return [self.decoders[i](X[i]) for i in range(2)]
    
    def forward(self, X, A):
        output = Namespace()
        
        # A = torch.eye(A.shape[0]).to(A.device)
        
        gex_pos_z, gex_neg_z, gex_summary, pex_pos_z, pex_neg_z, pex_summary = self.encode(X, A)
        encoded = [gex_pos_z, pex_pos_z]
        zs, mus, logvars = self.refactor(encoded, A)
        combined = self.combine(zs)
        X_hat = self.decode(combined)
        output.adj_recon = self.adjacency(combined[0])      

        output.gex_z, output.pex_z = zs
        output.gex_pos_z = gex_pos_z
        output.pex_pos_z = pex_pos_z
        output.gex_neg_z = gex_neg_z
        output.pex_neg_z = pex_neg_z
        output.gex_summary = gex_summary
        output.pex_summary = pex_summary
        output.gex_model_weight = self.encoders[0].weight
        output.pex_model_weight = self.encoders[1].weight
        output.gex_mu, output.pex_mu = mus
        output.gex_logvar, output.pex_logvar = logvars
        output.gex_c, output.pex_c = combined
        output.gex_recons, output.pex_recons = X_hat
        output.gex_input, output.pex_input = X

        return output
    
    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    
    @torch.no_grad()
    def impute(self, X, adj, enable_dropout=False, return_z=False):
        self.eval()
        
        # adj = torch.eye(adj.shape[0]).to(adj.device)
        
        
        
        if enable_dropout:
            self.enable_dropout()
        edge_index = adj.nonzero().t().contiguous()
        pos_z, _, _ = self.encoders[0](X, edge_index)
        z = self.fc_mus[0](pos_z)
        decoded = self.decoders[1](z)
        if return_z:
            return decoded.cpu().numpy(), z.cpu().numpy()       
            
        return decoded.cpu().numpy()
    
    
    # @torch.no_grad()
    # def impute(self, X, A, return_z=False):
    #     self.eval()
        
    #     pos_z = self.encoders[0](X)
    #     z = self.fc_mus[0](pos_z)
    #     decoded = self.decoders[1](z)
    #     if return_z:
    #         return decoded.cpu().numpy(), z.cpu().numpy() 
    #     return decoded.cpu().numpy()