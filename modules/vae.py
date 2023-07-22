from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.graph import GraphConvolution
from modules.inner_product import InnerProductDecoder
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JointVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        dropout=0,
        encoder_dim = 64,
        latent_dim=32,
    ):
        super().__init__()

        self.num_modalities = 2
        
        self.encoders = nn.ModuleList([
            GraphConvolution(
                input_dim=input_dim[0], 
                output_dim=encoder_dim, 
                dropout=dropout, 
                act=F.leaky_relu
            ),
            GraphConvolution(
                input_dim=input_dim[1], 
                output_dim=encoder_dim, 
                dropout=dropout, 
                act=F.leaky_relu
            )
        ])

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
        
        dim1, dim2 = input_dim
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
            )
        ])
        
        self.adjacency = InnerProductDecoder(
            dropout=dropout, 
            act=lambda x: x)

        ##  ω < 1 implies that modality 0 is weighted more
        self.omega = nn.Parameter(torch.rand(2))

    def _encode_gex(self, X, A):
        return self.encoders[0](X, A)
    
    def _encode_pex(self, X, A):
        return self.encoders[1](X, A)

    def encode(self, X, A):
        return self._encode_gex(X[0], A), self._encode_pex(X[1], A)

    def refactor(self, X, A):
        zs = []; mus = []; logvars = []
        for x, i in zip(X, range(2)):
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
                self.omega[i] * Z[i]
                + self.omega[(i + 1) % 2] * torch.mm(
                    corr if i == 0 else torch.t(corr),
                    Z[(i + 1) % 2])
            ) / (
                self.omega[i]
                + self.omega[(i + 1) % 2] * corr.sum((i + 1) % 2).reshape(-1, 1)
            )
            for i in range(self.num_modalities)
        ]

    def decode(self, X):
        return [self.decoders[i](X[i]) for i in range(self.num_modalities)]

    def forward(self, gene_matrix, protein_matrix, adjacency_matrix):
        output = Namespace()
        X = [gene_matrix, protein_matrix]
        A = adjacency_matrix
        corr = torch.eye(gene_matrix.shape[0], protein_matrix.shape[0]).to(device)
        
        encoded = self.encode(X, A)
        zs, mus, logvars = self.refactor(encoded, A)
        combined = self.combine(zs, corr)
        X_hat = self.decode(combined)
        output.adj_recon = self.adjacency(combined[0])      
        output.omega = self.omega  

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
    

