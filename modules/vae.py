from argparse import Namespace
from typing import Tuple
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
from modules.graph import GraphConvolution
from modules.inner_product import InnerProductDecoder
from modules.linear import LinearBlock
import numpy as np

class JointVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=32,
        dropout=0,
        encoder_dim = 64,
        latent_dim=32,
    ):
        super().__init__()

        self.num_modalities = 2
        
        if dropout is None:
            dropout = .6 if max(input_dim) > 64 else 0
        
        self.encoders = []
        for i in range(self.num_modalities):
            self.encoders.append(
                GraphConvolution(
                    input_dim=input_dim[i], 
                    output_dim=encoder_dim, 
                    dropout=dropout, 
                    act=F.leaky_relu
                )
            )
        self.encoders = nn.ModuleList(self.encoders)

        intermediate_dim = []
        for i in range(self.num_modalities):
            intermediate_dim.append(input_dim[i])


        self.fc_mus = nn.ModuleList([
            GraphConvolution(
                input_dim=encoder_dim, 
                output_dim=latent_dim, 
                dropout=dropout, 
                act=F.leaky_relu
            ),
            GraphConvolution(
                input_dim=encoder_dim, 
                output_dim=latent_dim, 
                dropout=dropout, 
                act=F.leaky_relu
            )
        ])

        self.fc_vars = nn.ModuleList([
            GraphConvolution(
                    input_dim=encoder_dim, 
                    output_dim=latent_dim, 
                    dropout=dropout, 
                    act=F.leaky_relu
            ),
            GraphConvolution(
                    input_dim=encoder_dim, 
                    output_dim=latent_dim, 
                    dropout=dropout, 
                    act=F.leaky_relu
            )
        ])

        self.decoders = []
        for i in range(self.num_modalities):
            self.decoders.append(nn.Sequential(
                nn.Linear(latent_dim, input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim[i], 2*input_dim[i]),
                nn.BatchNorm1d(2*input_dim[i]),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(2*input_dim[i], input_dim[i]),
            ))
        self.decoders = nn.ModuleList(self.decoders)
        
        self.adjacency = InnerProductDecoder(
            dropout=dropout, 
            act=lambda x: x)

        ## A value δ < 1 implies that modality 0 is weighted more than modality 1 during aggregation, 
        ## while δ > 1 implies the opposite. 
        self.sigma = nn.Parameter(torch.rand(self.num_modalities))

    def encode(self, X, A):
        return [self.encoders[i](X[i], A) for i in range(self.num_modalities)]

    def refactor(self, X, A, index=None):
        if index is None:
            index = range(self.num_modalities)
        zs = []; mus = []; logvars = []
        for x, i in zip(X, index):
            mu = self.fc_mus[i](x, A)
            logvar = self.fc_vars[i](x, A)
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

    def combine(self, Z, corr):
        """This moves correspondent latent embeddings 
        in similar directions over the course of training 
        and is key in the formation of similar latent spaces.""" 
        
        return [
            (
                self.sigma[i] * Z[i]
                + self.sigma[(i + 1) % 2] * torch.mm(
                    corr if i == 0 else torch.t(corr),
                    Z[(i + 1) % 2])
            ) / (
                self.sigma[i]
                + self.sigma[(i + 1) % 2] * corr.sum((i + 1) % 2).reshape(-1, 1)
            )
            for i in range(self.num_modalities)
        ]

    def decode(self, X):
        return [self.decoders[i](X[i]) for i in range(self.num_modalities)]

    def forward(self, X, A):
        output = Namespace()
        corr = torch.eye(X[0].shape[0], X[1].shape[0])
        
        encoded = self.encode(X, A)
        zs, mus, logvars = self.refactor(encoded, A)
        combined = self.combine(zs, corr)
        X_hat = self.decode(combined)
        output.adj_recon = self.adjacency(combined[0])      
        output.sigma = self.sigma  

        output.gex_z, output.pex_z = zs
        output.gex_mu, output.pex_mu = mus
        output.gex_logvar, output.pex_logvar = logvars
        output.gex_c, output.pex_c = combined
        output.gex_recons, output.pex_recons = X_hat
        output.gex_input, output.pex_input = X

        return output
    
    @torch.no_grad()
    def swap_latent(self, source, adj, from_modality=0, to_modality=1):
        self.eval()
        encoded_source = self.encoders[from_modality](source, adj)
        z = self.fc_mus[from_modality](encoded_source, adj)
        decoded = self.decoders[to_modality](z)
        return decoded.cpu().numpy()       
    

