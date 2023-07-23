from argparse import Namespace
import torch
import torch.nn.functional as F
import numpy as np
from math import prod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_distance(a, b, dist_method='cosine'):
    if dist_method == 'cosine':
        # Cosine Similarity
        sim = (
            torch.mm(a, torch.t(b))
            / (a.norm(dim=1).reshape(-1, 1) * b.norm(dim=1).reshape(1, -1))
        )
        diff = 1 - sim
        return sim, diff

    elif dist_method == 'euclidean':
        # Euclidean Distance
        dist = torch.cdist(a, b, p=2)
        sim = 1 / (1+dist)
        sim = 2 * sim - 1  # This scaling line is important
        sim = sim / sim.max()
        return sim, dist

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


_f = lambda x: float(x)
class Metrics:
    def __init__(self, track=False):
        self.track = track
        self.means = Namespace()
        self.counters = {}
        self.values = Namespace()
        self.losses = []
        
    def __str__(self) -> str:
        str_repr = ''
        for k, v in self.means.__dict__.items():
            str_repr+=f'{k}: {v:.3e}'
            str_repr+=' | '
            
        str_repr+=f'loss: {np.mean(self.losses):.3e}'
        return str_repr
    
    def __repr__(self) -> str:
        return self.__str__()  
    
    def custom_repr(self, keys):
        ...  
    
    def update_value(self, name, value):
        ... 

        
    def update(self, ledger):
        loss = 0
        _means = self.means.__dict__
        _values = self.values.__dict__
        
        for name, value in ledger.__dict__.items(): 
            if 'loss' not in name:
                continue
            loss += value
            if name not in _means:
                _means[name] = _f(value)
                self.counters[name] = 0
                if self.track:
                    _values[name] = [_f(value)]
            else:
                _means[name] = (_f(
                    (_means[name]*self.counters[name]) + _f(value)
                )) / (self.counters[name] + 1)
                if self.track:
                    _values[name].append(_f(value))
            self.counters[name] += 1
        self.losses.append(_f(loss))
        
        return loss
    
    def __call__(self, ledger):
        return self.update(ledger)
    
    def get(self, name):
        return self.mean.__dict__.get(name, None)

    def get_values(self, name):
        return self.values.__dict__.get(name, None)
    
    
class LossFunctions:

    @staticmethod
    def binary_cross_entropy(preds, labels, pos_weight, norm):
        cost = norm * F.binary_cross_entropy_with_logits(
            preds, 
            labels, 
            pos_weight=pos_weight,
            reduction='mean')
        return cost    
    
    @staticmethod
    def spatial_loss(z1, z2, sp_dists):
        """
        Pushes the closeness between embeddings to 
        not only reflect the expression similarity 
        but also their spatial proximity
        """
        sp_loss = 0
        for z in [z1, z2]:
            z_dists = torch.cdist(z, z, p=2)
            z_dists = torch.div(z_dists, torch.max(z_dists))
            n_items = z.size(dim=0) * z.size(dim=0)
            sp_loss += torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items)
        return sp_loss 
        
    @staticmethod
    def alignment_loss(emb1, emb2, P):
        """
        Pushes the embeddings for each modality 
        to be on the same latent space.
        """
        P_sum = P.sum(axis=1)
        P_sum[P_sum==0] = 1
        P = P / P_sum[:, None]
        _, cdiff = compute_distance(emb1, emb2)
        weighted_P_cdiff = cdiff * P
        alignment_loss = weighted_P_cdiff.absolute().sum() / P.absolute().sum()
        return alignment_loss

    @staticmethod
    def balance_loss(sigma):
        """Balances the contribution of each modality 
        in shaping the shared latent space."""
        sig_norm = sigma / sigma.sum()
        sigma_loss = (sig_norm - .5).square().mean()
        return sigma_loss
    
    @staticmethod
    def mean_sq_error(reconstructed, data):
        """Controls the ability of the latent space to encode information."""
        reconstruction_loss = (reconstructed - data).square().mean(axis=1).mean(axis=0)
        return reconstruction_loss
    
    @staticmethod
    def cross_loss(comb1, comb2, W):
        _, comdiff1 = compute_distance(comb1, comb2)
        cross_loss = comdiff1 * W
        cross_loss = cross_loss.sum() / prod(W.shape)
        return cross_loss

    @staticmethod
    def kl(epoch, epochs, mu, logvar):
        """
        Controls the smoothness of the latent space.
        The KL divergence measures the amount of information lost 
        when using Q to approximate P.
        """
        kl_loss =  -.5 * torch.mean(
                        1
                        + logvar
                        - mu.square()
                        - logvar.exp(),
                        axis=1
                    ).mean(axis=0)
        
        c = epochs / 2  # Midpoint
        kl_anneal = 1 / ( 1 + np.exp( - 5 * (epoch - c) / c ) )
        kl_loss = 1e-3 * kl_anneal * kl_loss
        
        return kl_loss
    
    @staticmethod
    def f_recons(comb1, comb2):
        corF = torch.eye(comb1.shape[0], comb2.shape[0]).to(device)
        F_est = torch.square(
                    comb1 - torch.mm(corF, comb2)
                ).mean(axis=1).mean(axis=0)
        
        return F_est
    
    @staticmethod
    def cosine_loss(emb, comb):
        _, codiff = compute_distance(emb, comb)
        cosine_loss = torch.diag(codiff.square()).mean(axis=0) / emb.shape[1]
        return cosine_loss
    
    @staticmethod
    def contrastive_loss(z1, z2, tau: float = 0.5):
        def semi_loss(z1, z2, tau):
            fexp = lambda x: torch.exp(x / tau)
            refl_sim = fexp(sim(z1, z1))
            between_sim = fexp(sim(z1, z2))
            return -torch.log(
                between_sim.diag()
                / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

        l1 = semi_loss(z1, z2, tau)
        l2 = semi_loss(z1, z2, tau)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret


# class Annealer:
#     def __init__(self, max_epochs):
#         self.max_epochs = max_epochs
        
#     def __call__(self, epoch, beta):
#         c = self.max_epochs / 2 
#         kl_anneal = 1 / ( 1 + np.exp( - beta * (epoch - c) / c ) )
#         kl_loss = 1e-3 * kl_anneal * kl_loss




class Loss:
    
    _base_alpha = 1e-3
    
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs        
        self.alpha = {
            'kl_pex': self._base_alpha,
            'recons_pex': self._base_alpha,
            'cosine_gex': self._base_alpha,
            'cosine_pex': self._base_alpha,
            'consistency': self._base_alpha,
            'adj': self._base_alpha,
            'spatial': self._base_alpha,
            'alignment': self._base_alpha,
            'balance': self._base_alpha,
            'cross': self._base_alpha,
            'contrast': self._base_alpha
        }


    def __call__(self, epoch, varz):
        return self.compute(epoch, varz)
    
    def compute(self, epoch, varz) -> Namespace:
        ledger = Namespace()
        a = self.alpha
        ledger.kl_loss_pex = a['kl_pex'] * LossFunctions.kl(epoch, self.max_epochs, varz.pex_mu, varz.pex_logvar)
        ledger.recons_loss_pex = a['recons_pex'] * LossFunctions.mean_sq_error(varz.pex_recons, varz.pex_input)
        ledger.cosine_loss_gex = a['cosine_gex'] * LossFunctions.cosine_loss(varz.gex_z, varz.gex_c)
        ledger.cosine_loss_pex = a['cosine_pex'] * LossFunctions.cosine_loss(varz.pex_z, varz.pex_c)
        ledger.consistency_loss = a['consistency'] * LossFunctions.f_recons(varz.gex_c, varz.pex_c)
        ledger.adj_loss = a['adj'] * LossFunctions.binary_cross_entropy(varz.adj_recon, varz.adj_label, varz.pos_weight, varz.norm)
        ledger.spatial_loss = a['spatial'] * LossFunctions.spatial_loss(varz.gex_z, varz.pex_z, varz.gex_sp_dist)
        ledger.alignment_loss = a['alignment'] * LossFunctions.alignment_loss(varz.gex_z, varz.pex_z, varz.corr)
        ledger.cross_loss = a['cross'] * LossFunctions.cross_loss(varz.gex_c, varz.pex_c, varz.corr)
        ledger.sigma_loss = a['balance'] * LossFunctions.balance_loss(varz.omega)
        ledger.contrastive_loss = a['contrast'] * LossFunctions.contrastive_loss(varz.z1, varz.z2)
                
        return ledger
    
