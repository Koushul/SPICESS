from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.graph import GraphConvolution
from modules.inner_product import InnerProductDecoder
import numpy as np
from modules.linear import LinearBlock
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj
from modules.grace import Encoder, GRACE, drop_feature
from torch import Tensor as T
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JointVAE(nn.Module):
    def __init__(
        self,
        num_genes,
        num_proteins,
        dropout = 0,
        encoder_dim = 64,
        latent_dim = 32,
        num_layers = 2,
        tau = 0.5
    ):
        super().__init__()

        self.num_modalities = 2
        
        _gcn = Encoder(
            num_genes, 
            latent_dim, 
            F.leaky_relu, 
            base_model=GCNConv, 
            k=num_layers)
                
        self.gex_encoder = GRACE(
            encoder=_gcn, 
            num_hidden=latent_dim, 
            latent_dim=latent_dim, 
            tau=tau
        )
        
        self.pex_encoder = LinearBlock(
                input_dim=num_proteins, 
                output_dim=encoder_dim, 
                dropout=dropout, 
        )

        self.mean = LinearBlock(
            input_dim=encoder_dim, 
            output_dim=latent_dim, 
            dropout=dropout, 
        )

        self.var = LinearBlock(
            input_dim=encoder_dim, 
            output_dim=latent_dim, 
            dropout=dropout, 
        )
        
        self.pex_decoder = LinearBlock(
            input_dim=latent_dim, 
            output_dim=num_proteins, 
            dropout=dropout
        )
        
        self.adjacency = InnerProductDecoder(
            dropout=dropout, 
            act=lambda x: x,
            latent_dim=latent_dim
        )

        ##  ω < 1 implies that modality 0 is weighted more
        self.omega = nn.Parameter(torch.rand(2))

    
    def refactor(self, X) -> tuple[T, T, T]:
        """Reparameterization trick to sample protein_z from N(mu, var)"""
        mu = self.mean(X)
        logvar = self.var(X)
        std = torch.exp(logvar / 2)
        if not self.training:
            z = mu
        else:
            std = std + 1e-7
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

        return z, mu, logvar

    def combine(self, Z, corr) -> list[T, T]:
        """Moves correspondent latent embeddings 
        in similar directions over the course of training 
        and is key in the formation of similar latent spaces""" 
        
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


    def contrast_embeddings(self, gene_matrix, edge_index, extra_params) -> tuple[T, T]:
        """Generates two views of the gene expression graph"""
        edge_index_1 = dropout_adj(edge_index, p=extra_params.drop_edge_rate_1)[0]
        edge_index_2 = dropout_adj(edge_index, p=extra_params.drop_edge_rate_2)[0]
        x_1 = drop_feature(gene_matrix, extra_params.drop_feature_rate_1)
        x_2 = drop_feature(gene_matrix, extra_params.drop_feature_rate_2)
        z1 = self.gex_encoder(x_1, edge_index_1)
        z2 = self.gex_encoder(x_2, edge_index_2)
        
        z1 = self.gex_encoder.projection(z1)
        z2 = self.gex_encoder.projection(z2)
        
        return z1, z2
        
    

    def forward(self, gene_matrix, protein_matrix, adjacency_matrix, extra_params):
        
        ## Housekeeping
        output = Namespace()
        A = adjacency_matrix
        edge_index = A.nonzero().t().contiguous()
        corr = torch.eye(gene_matrix.shape[0], protein_matrix.shape[0]).to(device)
        
        ## st_gene_expression -> ~z1, ~z2
        z1, z2 = self.contrast_embeddings(gene_matrix, edge_index, extra_params)
        
        ## st_gene_expression -> latent_space 
        gex_z = self.gex_encoder(gene_matrix, edge_index)
        
        ## protein_expression -> latent_space 
        encoded_pex = self.pex_encoder(protein_matrix)
        pex_z, mu, logvar = self.refactor(encoded_pex)
        
        ## Align latent spaces        
        combined = self.combine([gex_z, pex_z], corr)
        
        ## latent_space -> protein_expression
        pex_recons = self.pex_decoder(combined[1])
        
        ## latent_space -> connected_graph
        adj_recons = self.adjacency(combined[0]) 
        
        w = self.omega
        
        ## Output
        output.gex_z, output.pex_z = gex_z, pex_z
        output.z1, output.z2 = z1, z2
        output.adj_recon = adj_recons     
        output.omega = w
        output.pex_mu = mu
        output.pex_logvar = logvar
        output.gex_c, output.pex_c = combined
        output.pex_recons = pex_recons
        output.gex_input = gene_matrix
        output.pex_input = protein_matrix

        return output
    
    @torch.no_grad()
    def swap_latent(self, gene_matrix, A):
        self.eval()
        return self.pex_decoder(
            self.gex_encoder(gene_matrix, A.nonzero().t().contiguous())
        ).detach().cpu().numpy()
    

