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


class GraphVAE(nn.Module):   
    def __init__(self, 
            input_feat_dim: int, 
            encoder_dim: int, 
            latent_dim: int, 
            decoder_dim: int, 
            dropout: int) -> None:
        
        super(GVAE, self).__init__()
        
        meanMin = 1e-5
        meanMax = 1e6
        thetaMin = 1e-5
        thetaMax = 1e6
        
        self.gc = GraphConvolution(
            input_dim=input_feat_dim, 
            output_dim=encoder_dim, 
            dropout=dropout, 
            act=F.leaky_relu) 
        
        self.gc_mu = GraphConvolution(
            input_dim=encoder_dim, 
            output_dim=latent_dim, 
            dropout=dropout, 
            act=F.leaky_relu) 
        
        self.gc_logvar = GraphConvolution(
            input_dim=encoder_dim, 
            output_dim=latent_dim, 
            dropout=dropout, 
            act=F.leaky_relu) 
            
        self.adjacency = InnerProductDecoder(
            dropout=dropout, 
            act=lambda x: x)
    
        self.latent = LinearBlock(
            input_dim = latent_dim, 
            output_dim = decoder_dim, 
            dropout = dropout, 
            act = F.leaky_relu, 
            batchnorm = True)
        
        self.pi = LinearBlock(
            input_dim=decoder_dim, 
            output_dim=input_feat_dim, 
            dropout=0, 
            act = torch.sigmoid, 
            batchnorm = False,
            bias = True) 
        
        self.theta = LinearBlock(
            input_dim = decoder_dim, 
            output_dim = input_feat_dim, 
            dropout = 0, 
            act = lambda x: torch.clamp(
                        input=F.softplus(x), 
                        min=thetaMin, 
                        max=thetaMax),
            batchnorm = False,
            bias = True)

        self.mean = LinearBlock(
            input_dim = decoder_dim, 
            output_dim = input_feat_dim, 
            dropout = 0, 
            act = lambda x: torch.clamp(
                        input=torch.exp(input=x), 
                        min=meanMin, 
                        max=meanMax),
            batchnorm = False,
            bias = True)

    def dry_run(self, x=None, adj=None):
        self.eval()
        with torch.no_grad():
            if x is None:
                x = torch.rand(3, self.gc.weight.shape[0]).cuda()
            if adj is None:
                adj = torch.rand(3, 3).float().cuda()
            
            try:
                self.forward(x, adj)
                self.train()
                print("✅ Passed")
            except Exception as e:
                self.train()
                print("❌ Failed")
            
    def encode(self, x, adj) -> Tuple[FloatTensor, FloatTensor]:
        """
        The encoder network outputs the mean mu and log variance logvar of the latent variable distribution z.
        """
        graph_embedding = self.gc(x, adj) ## Graph Convolution
        return self.gc_mu(graph_embedding, adj), self.gc_logvar(graph_embedding, adj)

    def reparameterize(self, mu: FloatTensor, logvar: FloatTensor) -> FloatTensor:
        """
        The reparameterization trick is used to make the gradients flow through the sampling operation, 
        which is otherwise non-differentiable. 
        """
        std = torch.exp(logvar / 2)
        std = std + 1e-7
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z
    

    def decode(self, z: FloatTensor) -> Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor]:
        """
        The decoder network outputs the mean and dispersion parameters of the negative binomial distribution, 
        and the probability of the point mass at zero. 
        The mean and dispersion parameters are used to model the count data, 
        while the probability of the point mass at zero is used to model the excess zeros.
        """
        o = self.latent(z)
        pi = self.pi(o)
        theta = self.theta(o)
        mean = self.mean(o)
        return o, pi, theta, mean
    
    def forward(self, x: FloatTensor, adj: LongTensor) -> Namespace:
        output = Namespace()
        output.mu, output.logvar = self.encode(x, adj)
        output.z = self.reparameterize(mu=output.mu, logvar=output.logvar)
        output.adj_recon = self.adjacency(output.z)
        output.o, output.pi, output.theta, output.mean = self.decode(z=output.z)
        output.features_recon = [output.o, output.pi, output.theta, output.mean]
        return output
        
class GVAE(nn.Module):   
    def __init__(self, 
            input_feat_dim: int, 
            encoder_dim: int, 
            latent_dim: int, 
            decoder_dim: int, 
            dropout: int) -> None:
        
        super(GVAE, self).__init__()
        
        meanMin = 1e-5
        meanMax = 1e6
        
        self.gc = GraphConvolution(
            input_dim=input_feat_dim, 
            output_dim=encoder_dim, 
            dropout=dropout, 
            act=F.leaky_relu) 
        
        self.gc_mu = GraphConvolution(
            input_dim=encoder_dim, 
            output_dim=latent_dim, 
            dropout=dropout, 
            act=F.leaky_relu) 
        
        self.gc_logvar = GraphConvolution(
            input_dim=encoder_dim, 
            output_dim=latent_dim, 
            dropout=dropout, 
            act=F.leaky_relu) 
        

        self.adjacency = InnerProductDecoder(
            dropout=dropout, 
            act=lambda x: x)
    
        self.linear_latent = LinearBlock(
            input_dim = latent_dim, 
            output_dim = decoder_dim, 
            dropout = dropout, 
            act = F.leaky_relu, 
            batchnorm = True)
        
        self.reconstructor = LinearBlock(
            input_dim = decoder_dim, 
            output_dim = input_feat_dim, 
            dropout = 0, 
            act = lambda x: torch.clamp(
                        input=torch.exp(input=x), 
                        min=meanMin, 
                        max=meanMax),
            batchnorm = False,
            bias = True)
        

    def encode(self, x, adj) -> Tuple[FloatTensor, FloatTensor]:
        graph_embedding = self.gc(x, adj) ## Graph Convolution
        return self.gc_mu(graph_embedding, adj), self.gc_logvar(graph_embedding, adj)
    

    def reparameterize(self, mu: FloatTensor, logvar: FloatTensor):
        std = torch.exp(logvar / 2)
        if not self.training:
            z = mu
        else:
            std = std + 1e-7
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

        return z
    
    def decode(self, z: FloatTensor) -> Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor]:
        hidden = self.linear_latent(z)
        reconstructed = self.reconstructor(hidden)
        return reconstructed
    
    def forward(self):
        raise NotImplementedError("This method is not implemented")
    
    
class JointVAE(nn.Module):   
    def __init__(self, 
            num_genes: int, 
            num_proteins:int,
            latent_dim: int, 
            gex_encoder_dim:int, 
            gex_decoder_dim:int, 
            pex_encoder_dim:int, 
            pex_decoder_dim:int,
            gex_dropout:int=0,
            pex_dropout:int=0) -> None:
        
        super(JointVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.gene_model = GVAE(
            input_feat_dim=num_genes,
            encoder_dim=gex_encoder_dim, 
            latent_dim=latent_dim, 
            decoder_dim=gex_decoder_dim, 
            dropout=gex_dropout)
        
        self.protein_model = GVAE(
            input_feat_dim=num_proteins,  
            encoder_dim=pex_encoder_dim, 
            latent_dim=latent_dim, 
            decoder_dim=pex_decoder_dim, 
            dropout=pex_dropout)
        
        self.sigma = nn.Parameter(torch.rand(2), requires_grad=True)
        
        
    def train_mode(self):
        self.train()
        self.gene_model.train()
        self.protein_model.train()

    def eval_mode(self):
        self.eval()
        self.gene_model.eval()
        self.protein_model.eval()

    @torch.no_grad()
    def dry_run(self, n=3, results=False):
        self.eval_mode()
        x1 = torch.rand(n, self.gene_model.gc.weight.shape[0]).cuda()
        x2 = torch.rand(n, self.protein_model.gc.weight.shape[0]).cuda()
        adj = torch.randint(0, 1, (n, n)).float().cuda()
        
        try:
            output = self.forward(x1, x2, adj)
            assert output.gex_adj_recon.shape == adj.shape
            assert output.pex_adj_recon.shape == adj.shape
            assert output.gex_recons.shape == x1.shape
            assert output.pex_recons.shape == x2.shape
            assert output.gex_z.shape == (n, self.latent_dim)
            assert output.pex_z.shape == (n, self.latent_dim)
            assert output.gex_c.shape == (n, self.latent_dim)
            assert output.pex_c.shape == (n, self.latent_dim)
            self.train_mode()
            print("✅ Passed")
            if results:
                return output
        except Exception as e:
            print("❌ Failed")
            # raise e
            print(e)
            
            
    def encode(self, gex: FloatTensor, pex: FloatTensor, adj: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        """
        Embed the gene matrix and protein matrix into a shared latent space,
        """
        gex_mu, gex_logvar = self.gene_model.encode(gex, adj)
        pex_mu, pex_logvar = self.protein_model.encode(pex, adj)
        return gex_mu, gex_logvar, pex_mu, pex_logvar
    
    def reparameterize(self, output) -> FloatTensor:
        gex_z = self.gene_model.reparameterize(output.gex_mu, output.gex_logvar) ## latent gene space
        pex_z = self.protein_model.reparameterize(output.pex_mu, output.pex_logvar) ## latent protein space
        return gex_z, pex_z
    
    def sync(self, X, corr):
        """
        Move corresponding latent embeddings in similar directions during training.
        This incentivizes the formation of similar latent spaces and cell-state representations.
        """
        return [
            (
                self.sigma[i] * X[i]
                + self.sigma[(i + 1) % 2] * torch.mm(
                    corr if i == 0 else torch.t(corr),
                    X[(i + 1) % 2])
            ) / (
                self.sigma[i]
                + self.sigma[(i + 1) % 2] * corr.sum((i + 1) % 2).reshape(-1, 1)
            )
            for i in range(len(X))
        ]

    def decode(self, gex_c: FloatTensor, pex_c: FloatTensor):        
        gex_recons = self.gene_model.decode(gex_c)
        pex_recons = self.protein_model.decode(pex_c)

        return gex_recons, pex_recons
    
    def forward(self, gex: FloatTensor, pex: FloatTensor, adj: LongTensor) -> Namespace:
        output = Namespace()
        corr = torch.eye(gex.shape[0], pex.shape[0]).float().cuda()
        output.gex_mu, output.gex_logvar, output.pex_mu, output.pex_logvar = self.encode(gex, pex, adj)
        output.gex_z, output.pex_z = self.reparameterize(output)
        output.gex_c, output.pex_c = self.sync([output.gex_z, output.pex_z], corr)
        output.gex_recons, output.pex_recons = self.decode(gex_c=output.gex_c, pex_c=output.pex_c)
        
        output.gex_adj_recon = self.gene_model.adjacency(output.gex_c)
        output.pex_adj_recon = self.protein_model.adjacency(output.pex_c)
        
        return output
    
class NonSpatialVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dropout=None,
    ):
        super().__init__()

        self.num_modalities = 2
        
        if dropout is None:
            dropout = .6 if max(input_dim) > 64 else 0

        self.encoders = []
        for i in range(self.num_modalities):
            self.encoders.append(nn.Sequential(
                nn.Linear(input_dim[i], 2*input_dim[i]),
                nn.BatchNorm1d(2*input_dim[i]),
                nn.LeakyReLU(),
                nn.Dropout(dropout),

                nn.Linear(2*input_dim[i], input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
            ))
        self.encoders = nn.ModuleList(self.encoders)

        intermediate_dim = []
        for i in range(self.num_modalities):
            intermediate_dim.append(input_dim[i])

        self.fc_mus = []
        for i in range(self.num_modalities):
            self.fc_mus.append(nn.Linear(intermediate_dim[i], output_dim))
        self.fc_mus = nn.ModuleList(self.fc_mus)

        self.fc_vars = []
        for i in range(self.num_modalities):
            self.fc_vars.append(nn.Linear(intermediate_dim[i], output_dim))
        self.fc_vars = nn.ModuleList(self.fc_vars)

        self.decoders = []
        for i in range(self.num_modalities):
            self.decoders.append(nn.Sequential(
                # UnionCom Like
                nn.Linear(output_dim, input_dim[i]),
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

        self.sigma = nn.Parameter(torch.rand(self.num_modalities))

    def encode(self, X):
        return [self.encoders[i](X[i]) for i in range(self.num_modalities)]

    def refactor(self, X, index=None):
        if index is None:
            index = range(self.num_modalities)
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

    def combine(self, X, corr):
        return [
            (
                self.sigma[i] * X[i]
                + self.sigma[(i + 1) % 2] * torch.mm(
                    corr if i == 0 else torch.t(corr),
                    X[(i + 1) % 2])
            ) / (
                self.sigma[i]
                + self.sigma[(i + 1) % 2] * corr.sum((i + 1) % 2).reshape(-1, 1)
            )
            for i in range(self.num_modalities)
        ]

    def decode(self, X):
        return [self.decoders[i](X[i]) for i in range(self.num_modalities)]

    def forward(self, *X, corr):
        output = Namespace()
        zs, mus, logvars = self.refactor(self.encode(X))
        output.gex_z, output.pex_z = zs
        output.gex_mu, output.pex_mu = mus
        output.gex_logvar, output.pex_logvar = logvars
        combined = self.combine(zs, corr)
        output.gex_c, output.pex_c = combined
        X_hat = self.decode(combined)
        output.gex_recons, output.pex_recons = X_hat
        # return zs, combined, X_hat, mus, logvars
        return output
    
class SpatialVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
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

        self.fc_mus = []
        for i in range(self.num_modalities):
            self.fc_mus.append(GraphConvolution(
            input_dim=encoder_dim, 
            output_dim=latent_dim, 
            dropout=dropout, 
            act=F.leaky_relu))
        self.fc_mus = nn.ModuleList(self.fc_mus)

        self.fc_vars = []
        for i in range(self.num_modalities):
            self.fc_vars.append(GraphConvolution(
                input_dim=encoder_dim, 
                output_dim=latent_dim, 
                dropout=dropout, 
                act=F.leaky_relu))
        self.fc_vars = nn.ModuleList(self.fc_vars)

        self.decoders = []
        for i in range(self.num_modalities):
            self.decoders.append(nn.Sequential(
                nn.Linear(output_dim, input_dim[i]),
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

    def combine(self, X, corr):
        return [
            (
                self.sigma[i] * X[i]
                + self.sigma[(i + 1) % 2] * torch.mm(
                    corr if i == 0 else torch.t(corr),
                    X[(i + 1) % 2])
            ) / (
                self.sigma[i]
                + self.sigma[(i + 1) % 2] * corr.sum((i + 1) % 2).reshape(-1, 1)
            )
            for i in range(self.num_modalities)
        ]

    def decode(self, X):
        return [self.decoders[i](X[i]) for i in range(self.num_modalities)]

    def forward(self, X, A, corr):
        output = Namespace()
        zs, mus, logvars = self.refactor(self.encode(X, A), A)
        output.gex_z, output.pex_z = zs
        output.gex_mu, output.pex_mu = mus
        output.gex_logvar, output.pex_logvar = logvars
        combined = self.combine(zs, corr)
        output.gex_c, output.pex_c = combined
        X_hat = self.decode(combined)
        output.gex_recons, output.pex_recons = X_hat
        output.adj_recon = self.adjacency(combined[0])        
        return output
    
    @torch.no_grad()
    def swap_latent(self, source, adj, from_modality=0, to_modality=1):
        self.eval()
        encoded_source = self.encoders[from_modality](source, adj)
        z = self.fc_mus[from_modality](encoded_source, adj)
        decoded = self.decoders[to_modality](z)
        return decoded        
