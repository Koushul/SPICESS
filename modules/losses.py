from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

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

    
