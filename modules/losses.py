from argparse import Namespace
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from math import prod
class LossFunctions:
    
    def __init__(self, reconWeight=300, ridgePi=0.05):
        self.reconWeight = reconWeight
        self.ridgePi = ridgePi
        self.sp = []
        self.kl = []
        self.ce = []
        self.zinb = []
        self.loss = []
        
    def SP(self, z, sp_dists):
        """            
        Args:
        - z: tensor of shape (batch_size, latent_dim) representing the latent variables.
        - sp_dists: tensor of shape (batch_size, batch_size) representing the shortest path distances between nodes.
        """
        z_dists = torch.cdist(z, z, p=2)
        z_dists = torch.div(z_dists, torch.max(z_dists))
        n_items = z.size(dim=0) * z.size(dim=0)
        p = torch.div(torch.sum(torch.mul(1.0 - z_dists.cuda(), sp_dists.cuda())), n_items).cuda()
        self.sp.append(float(p))
        return p
    
    
    def KL(self, mu, logvar, nodemask=None, reduction='mean'):
        """
        Calculates the KL divergence between the latent distribution and the prior distribution.
        Given two probability distributions P and Q, 
        the KL divergence from P to Q measures the amount of information lost when Q is used to approximate P.

        Args:
        - mu: tensor of shape (batch_size, latent_dim) representing the mean of the latent distribution.
        - logvar: tensor of shape (batch_size, latent_dim) representing the log variance of the latent distribution.
        - nodemask: tensor of shape (batch_size,) representing the mask for the nodes.
        - reduction: string representing the reduction method for the KL divergence.

        """
        if reduction == 'mean':
            f = torch.mean
            if nodemask is None:
                s = mu.size()[0]
            else:
                s = nodemask.size()[0]
        
        elif reduction == 'sum':
            f = torch.sum
            s = 1
        
        if nodemask is None:
            kl = -(0.5 / s) * torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1)
            return kl
        
        kl = -(0.5 / s) * f(torch.sum(1 + 2 * logvar[nodemask] - mu[nodemask].pow(2) - logvar[nodemask].exp().pow(2), 1))

        self.kl.append(float(kl))
        return kl
    
    
    
    def CE(self, preds, labels, pos_weight, norm, nodemask=None):
        """
        Calculates the cross-entropy loss between the predicted and true adjacency matrices.
        The pos_weight argument is used to weight the positive class, 
        and the norm argument is used to normalize the loss by the number of nodes in the graph.
        
        Args:
        - preds: tensor of shape (batch_size, batch_size) representing the predicted adjacency matrix.
        - labels: tensor of shape (batch_size, batch_size) representing the true adjacency matrix.
        - pos_weight: float representing the weight of the positive class.
        - norm: float representing the normalization factor.
        - nodemask: tensor of shape (batch_size,) representing the mask for the nodes.
        
        Returns:
        - cost: tensor representing the cross-entropy loss.
        """
        if nodemask is None:
            cost = norm * F.binary_cross_entropy_with_logits(
                preds, 
                labels, 
                pos_weight=pos_weight,
                reduction='mean')
            return cost
        
        cost = norm * F.binary_cross_entropy_with_logits(
            preds[nodemask,:][:,nodemask], 
            labels[nodemask,:][:,nodemask], 
            pos_weight=pos_weight,
            reduction='mean')
        
        self.ce.append(float(cost))
        return cost

    def MSE(self, preds, inputs, mask, reconWeight, mse):
        cost = mse(preds[mask], inputs[mask])*reconWeight
        return cost


    def NB(self, preds, y_true, mask, reconWeight, eps=1e-10, ifmean=True):
        """
        Calculates the negative binomial reconstruction loss between the predicted and true gene expression values.
        
        Args:
        - preds: tuple of tensors representing the predicted gene expression values, pi, theta, and y_pred.
        - y_true: tensor of shape (batch_size,) representing the true gene expression values.
        - mask: tensor of shape (batch_size,) representing the mask for the nodes.
        - reconWeight: float representing the weight of the reconstruction loss.
        - eps: float representing the epsilon value.
        - ifmean: boolean representing whether to calculate the mean or not.
        """
        output, pi, theta, y_pred = preds
        nbloss1 = torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
        nbloss2 = (theta + y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))
        nbloss = nbloss1 + nbloss2
        if ifmean: nbloss = torch.mean(nbloss[mask])*reconWeight
        else: nbloss = nbloss
        return nbloss

    def ZINB(self, preds, y_true, mask, y_true_raw, eps=1e-10):
        """
        Calculates the zero-inflated negative binomial reconstruction loss between the predicted and true gene expression values.
        
        Args:
        - preds: tuple of tensors representing the predicted gene expression values, pi, theta, and y_pred.
        - y_true: tensor of shape (batch_size,) representing the true gene expression values.
        - mask: tensor of shape (batch_size,) representing the mask for the nodes.
        - eps: float representing the epsilon value.
        """
        output, pi, theta, y_pred = preds
        reconWeight, ridgePi = self.reconWeight, self.ridgePi
        nb_case = self.NB(preds, y_true, mask, reconWeight, eps=1e-10, ifmean=False)- torch.log(pi+eps)
        zero_nb = torch.pow(theta/(theta+y_pred+eps), theta)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.lt(y_true_raw, 1), zero_case, nb_case)
        ridge = ridgePi*pi*pi
        result += ridge
        result = torch.mean(result[mask])
        zloss = result*reconWeight
        self.zinb.append(float(zloss))
        return zloss


class LossFunctionsv2:
    
    
    def compute_distance(self, a, b, dist_method='euclidean'):
        if dist_method == 'cosine':
            # Cosine Similarity
            sim = (
                torch.mm(a, torch.t(b))
                / (a.norm(dim=1).reshape(-1, 1) * b.norm(dim=1).reshape(1, -1))
            )
            # return sim, 1-sim
            diff = 1 - sim
            # sim[sim < 0] = 0
            # diff[diff < 0] = 0
            return sim, diff

        elif dist_method == 'euclidean':
            # Euclidean Distance
            dist = torch.cdist(a, b, p=2)
            sim = 1 / (1+dist)
            sim = 2 * sim - 1  # This scaling line is important
            sim = sim / sim.max()
            return sim, dist
       
    
    def binary_cross_entropy(self, preds, labels, pos_weight, norm):
        cost = norm * F.binary_cross_entropy_with_logits(
            preds, 
            labels, 
            pos_weight=pos_weight,
            reduction='mean')
        return cost    
    
    def spatial_loss(self, z, sp_dists):
        z_dists = torch.cdist(z, z, p=2)
        z_dists = torch.div(z_dists, torch.max(z_dists))
        n_items = z.size(dim=0) * z.size(dim=0)
        p = torch.div(torch.sum(torch.mul(1.0 - z_dists.cuda(), sp_dists.cuda())), n_items).cuda()
        return p 
        
    def alignment_loss(self, emb1, emb2, P):
        """
        Incentivizes the low-dimensional embeddings for each modality
        to be on the sam latent space.
        """
        P_sum = P.sum(axis=1)
        P_sum[P_sum==0] = 1
        P = P / P_sum[:, None]
        csim, cdiff = self.compute_distance(emb1, emb2)
        weighted_P_cdiff = cdiff * P
        alignment_loss = weighted_P_cdiff.absolute().sum() / P.absolute().sum()
        return alignment_loss

        
    def sigma_loss(self, sigma):
        sig_norm = sigma / sigma.sum()
        sigma_loss = (sig_norm - .5).square().mean()
        return sigma_loss
    

    def mean_sq_error(self, reconstructed, data):
        reconstruction_loss = (reconstructed - data).square().mean(axis=1).mean(axis=0)
        return reconstruction_loss
    
    def cross_loss(self, comb1, comb2, F):
        comsim1, comdiff1 = self.compute_distance(comb1, comb2)
        cross_loss = comdiff1 * F
        cross_loss = cross_loss.sum() / prod(F.shape)
        return cross_loss

    def kl(self, epoch, epochs, mu, logvar):
        kl_loss =  -.5 * torch.mean(
                        1
                        + logvar
                        - mu.square()
                        - logvar.exp(),
                        axis=1
                    ).mean(axis=0)
        
        c = epochs / 2  # Midpoint
        kl_anneal = 1 / ( 1 + np.exp( - 5 * (epoch - c) / c ) )
        kl_loss = 32 * 1e-3 * kl_anneal * kl_loss
        
        return kl_loss
    
    def f_recons(self, comb1, comb2):
        corF = torch.eye(comb1.shape[0], comb2.shape[0]).float().cuda()
        F_est = torch.square(
                    comb1 - torch.mm(corF, comb2)
                ).mean(axis=1).mean(axis=0)
        
        return F_est
    
    def cosine_loss(self, emb1, emb2, comb1, comb2):
        # Cosine Loss
        cosim0, codiff0 = self.compute_distance(emb1, comb1)
        cosim1, codiff1 = self.compute_distance(emb2, comb2)
        
        cosine_loss = (
            torch.diag(codiff0.square()).mean(axis=0) / emb1.shape[1]
            + torch.diag(codiff1.square()).mean(axis=0) / emb2.shape[1])
        
        return 32 * cosine_loss


class Lossv2(LossFunctionsv2):
    
        
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.history = Namespace()
        self.history.kl_gex = []
        self.history.kl_pex = []
        self.history.recons_gex = []
        self.history.recons_pex = []
        self.history.cosine = []
        self.history.cons = []
        self.history.spatial = []
        self.history.adj = [] 
        self.history.align = [] 
                  

    def _update_means(self):
        self.mean_kl_gex = np.mean(self.history.kl_gex)
        self.mean_kl_pex = np.mean(self.history.kl_pex)
        self.mean_recons_gex = np.mean(self.history.recons_gex)
        self.mean_recons_pex = np.mean(self.history.recons_pex)
        self.mean_cosine = np.mean(self.history.cosine)
        self.mean_spatial = np.mean(self.history.spatial)
        self.mean_adj = np.mean(self.history.adj)
        self.mean_align = np.mean(self.history.align)

    
    def compute(self, epoch, varz):
        
        kl_loss_gex = self.alpha * self.kl(epoch, varz.epochs, varz.gex_mu, varz.gex_logvar)
        kl_loss_pex = self.alpha * self.kl(epoch, varz.epochs, varz.pex_mu, varz.pex_logvar)
        self.history.kl_gex.append(float(kl_loss_gex))
        self.history.kl_pex.append(float(kl_loss_pex))
        
        recons_loss_gex = self.alpha * self.mean_sq_error(varz.gex_recons, varz.gex_features_pca)
        recons_loss_pex = self.alpha * self.mean_sq_error(varz.pex_recons, varz.pex_features_pca)
        self.history.recons_gex.append(float(recons_loss_gex))
        self.history.recons_pex.append(float(recons_loss_pex))
        
        cosine_loss = self.alpha * self.cosine_loss(varz.gex_z, varz.pex_z, varz.gex_c, varz.pex_c)
        self.history.cosine.append(float(cosine_loss))
        
        consistency_loss = self.alpha * self.f_recons(varz.gex_c, varz.pex_c)
        self.history.cons.append(float(consistency_loss))
        
        adj_loss = self.alpha * self.binary_cross_entropy(
            varz.adj_recon, varz.adj_label, 
            varz.pos_weight, varz.norm)
        self.history.adj.append(float(adj_loss))
        
        spatial_loss = self.alpha * self.spatial_loss(varz.gex_z, varz.gex_sp_dist)
        self.history.spatial.append(float(spatial_loss))
        
        alignment_loss = self.alpha * self.alignment_loss(varz.gex_z, varz.pex_z, varz.corr)
        self.history.align.append(float(alignment_loss))
        
        self._update_means()
        
        return kl_loss_gex + \
            kl_loss_pex +  \
            recons_loss_gex + \
            recons_loss_pex + \
            cosine_loss + \
            consistency_loss + adj_loss + spatial_loss
    
    
    

class Loss(LossFunctionsv2):
    
    gex = Namespace()
    pex = Namespace()
    gex.kl = []
    pex.kl = []
    gex.recons = []
    pex.recos = []
    cosine = []
    f_est = []
    
    def __init__(self, alpha=1):
        self.alpha = alpha
    
    def compute(self, 
                epoch, 
                varz, 
                # sigma
            ):
        
        kl_loss_gex = self.kl(epoch, varz.epochs, varz.gex_mu, varz.gex_logvar)
        kl_loss_pex = self.kl(epoch, varz.epochs, varz.pex_mu, varz.pex_logvar)
        
        self.gex.kl.append(float(kl_loss_gex))
        self.pex.kl.append(float(kl_loss_pex))
        
        recons_loss_gex = self.mean_sq_error(varz.gex_recons, varz.gex_features_pca)
        recons_loss_pex = self.mean_sq_error(varz.pex_recons, varz.pex_features_pca)
        
        self.gex.recons.append(float(recons_loss_gex))
        self.pex.recos.append(float(recons_loss_pex))
        
        cosine_loss = self.cosine_loss(varz.gex_z, varz.pex_z, varz.gex_c, varz.pex_c)
        F_est = self.f_recons(varz.gex_c, varz.pex_c)
        
        self.cosine.append(float(cosine_loss))
        self.f_est.append(float(F_est))
        
        # sigma_loss = self.sigma_loss(sigma)
        
        return self.alpha * (kl_loss_gex + \
            kl_loss_pex +  \
            recons_loss_gex + \
            recons_loss_pex + \
            cosine_loss + \
            F_est)
            # sigma_loss
    
    
    
    
class LossKZCP(LossFunctions):
    """
    Computes 
        - KL, 
        - ZINB, 
        - CE, 
        - SP
    """
    def compute(self, varz, kl_weight=1.0, recon_weight=1.0, adj_weight=1.0, sp_weight=1.0):        
        loss_kl = self.KL(varz.mu, varz.logvar, varz.train_nodes_idx) ## latent dist vs prior
        loss_x = self.ZINB(varz.features_recon, varz.feature, varz.train_nodes_idx, varz.features_raw) ## gene reconstruction 
        loss_a = self.CE(varz.adj_recon, varz.adj_label, varz.pos_weight, varz.norm, varz.train_nodes_idx) ## adj reconstruction
        loss_p = self.SP(varz.z, varz.sp_dists)
        
        loss = kl_weight*loss_kl + recon_weight*loss_x + adj_weight*loss_a + sp_weight*loss_p
        self.loss.append(float(loss))
        
        return loss

    
