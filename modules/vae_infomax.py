## Author: Koushul Ramjattun

from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.graph import GraphConvolution
from modules.inner_product import InnerProductDecoder
import numpy as np
from modules.infomax import DeepGraphInfomax
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

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
        encoder_dim = 64,
        latent_dim=32,
    ):
        super().__init__()

        self.num_modalities = 2
        
        self.encoders = nn.ModuleList([
            DeepGraphInfomax(
                hidden_channels=latent_dim, 
                encoder=GraphEncoder(input_dim[0], latent_dim),
                summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
                corruption=corruption),
            DeepGraphInfomax(
                hidden_channels=latent_dim, 
                encoder=GraphEncoder(input_dim[1], latent_dim),
                summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
                corruption=corruption),
        ])

        self.fc_mus = nn.ModuleList([
            GraphConvolution(
                input_dim=latent_dim, 
                output_dim=latent_dim, 
                dropout=dropout, 
                act=F.leaky_relu
            ),
            GraphConvolution(
                input_dim=latent_dim, 
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

        

    def encode(self, X, A):
        edge_index = A.nonzero().t().contiguous()
        gex_pos_z, gex_neg_z, gex_summary = self.encoders[0](X[0], edge_index)
        pex_pos_z, pex_neg_z, pex_summary = self.encoders[1](X[1], edge_index)                
        return [gex_pos_z, gex_neg_z, gex_summary, pex_pos_z, pex_neg_z, pex_summary]
        

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

    def combine(self, Z):
        """This moves correspondent latent embeddings 
        in similar directions over the course of training 
        and is key in the formation of similar latent spaces."""         
        mZ = 0.5*(Z[0] + Z[1])        
        return [mZ, mZ]


    def decode(self, X):
        return [self.decoders[i](X[i]) for i in range(self.num_modalities)]
    
    

    def forward(self, X, A):
        output = Namespace()
        # corr = torch.eye(X[0].shape[0], X[1].shape[0]).to(device)
        
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
    
    
    @torch.no_grad()
    def impute(self, X, adj):
        self.eval()
        edge_index = adj.nonzero().t().contiguous()
        pos_z, neg_z, summary = self.encoders[0](X, edge_index)
        z = self.fc_mus[0](pos_z, adj)
        decoded = self.decoders[1](z)
        return decoded.cpu().numpy()       
    

