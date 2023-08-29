from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.inner_product import InnerProductDecoder

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

class NonSpatialVAE(nn.Module):
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
                
        dim1, dim2 = input_dim
        
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim1, encoder_dim),
                nn.BatchNorm1d(encoder_dim),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                nn.Linear(dim2, encoder_dim),
                nn.BatchNorm1d(encoder_dim),
                nn.LeakyReLU(),
            ),
        ])
        
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
                

    def encode(self, X):
        enc_a = self.encoders[0](X[0])
        enc_b = self.encoders[1](X[1])                
        return [enc_a, enc_b]
        

    def refactor(self, X):
        index = range(2)
        zs = []; mus = []; logvars = []
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
    
    

    def forward(self, X):
        output = Namespace()        
        enc_g, enc_p = self.encode(X)
        encoded = [enc_g, enc_p]
        zs, mus, logvars = self.refactor(encoded)
        combined = self.combine(zs)
        X_hat = self.decode(combined)

        output.gex_z, output.pex_z = zs
        output.gex_pos_z = enc_g
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
    def impute(self, X, enable_dropout=False):
        self.eval()
        if enable_dropout:
            self.enable_dropout()
        enc = self.encoders[0](X)
        z = self.fc_mus[0](enc)
        decoded = self.decoders[1](z)
        return decoded.cpu().numpy()      