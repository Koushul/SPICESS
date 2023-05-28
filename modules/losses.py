import torch
import torch.nn.functional as F

class LossFunctions:
    
    def __init__(self, reconWeight, ridgePi, y_true_raw):
        self.reconWeight = reconWeight
        self.ridgePi = ridgePi
        self.y_true_raw = y_true_raw
        self.penalties = []
        self.kl = []
        self.ce = []
        self.zinb = []
        self.loss = []
        
        
    def plot(self):
        ...
        
        
    def compute(self, varz, reg=False):        
        loss_kl_train = self.KL(varz.mu, varz.logvar, varz.train_nodes_idx)
        loss_x_train = self.ZINB(varz.features_recon, varz.feature, varz.train_nodes_idx)
        loss_a_train = self.CE(varz.adj_recon, varz.adj_label, varz.pos_weight, varz.norm, varz.train_nodes_idx)
        loss = loss_kl_train + loss_x_train + loss_a_train

        if reg: 
            p = self.penalty(varz.z, varz.sp_dists)
            loss = loss + p
            self.penalties.append(float(p))
        
        self.loss.append(float(loss))
        return loss
    
    
    def penalty(self, z, sp_dists):
        z_dists = torch.cdist(z, z, p=2)
        z_dists = torch.div(z_dists, torch.max(z_dists))
        n_items = z.size(dim=0) * z.size(dim=0)
        p = torch.div(torch.sum(torch.mul(1.0 - z_dists.cuda(), sp_dists.cuda())), n_items).cuda()
        return p
    
    def KL(self, mu, logvar, nodemask=None, reduction='mean'):
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
            kl = -(0.5 / s) * flask(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
            return kl
        kl = -(0.5 / s) * f(torch.sum(1 + 2 * logvar[nodemask] - mu[nodemask].pow(2) - logvar[nodemask].exp().pow(2), 1))
        
        self.kl.append(float(kl))
        return kl
    
    
    def CE(self, preds, labels, pos_weight, norm, nodemask=None):
        if nodemask is None:
            cost = norm * F.binary_cross_entropy_with_logits(preds, 
                                                             labels, 
                                                             pos_weight=pos_weight,
                                                             reduction='mean')
            return cost
        
        cost = norm * F.binary_cross_entropy_with_logits(preds[nodemask,:][:,nodemask], 
                                                         labels[nodemask,:][:,nodemask], 
                                                         pos_weight=pos_weight,
                                                         reduction='mean')
        self.ce.append(float(cost))
        return cost

    def MSE(self, preds, inputs, mask, reconWeight, mse):
        cost = mse(preds[mask], inputs[mask])*reconWeight
        return cost


    def NB(self, preds, y_true, mask, reconWeight, eps=1e-10, ifmean=True):
        output, pi, theta, y_pred = preds
        nbloss1 = torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
        nbloss2 = (theta + y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))
        nbloss = nbloss1 + nbloss2
        if ifmean: nbloss = torch.mean(nbloss[mask])*reconWeight
        else: nbloss = nbloss
        return nbloss

    def ZINB(self, preds, y_true, mask, eps=1e-10):
        output, pi, theta, y_pred = preds
        reconWeight, ridgePi, y_true_raw = self.reconWeight, self.ridgePi, self.y_true_raw
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