import sys
sys.path.append('..')
from data_loaders import DataBlob
from utils import featurize, update_vars
import scanpy as sc
from modules.vae import SpatialVAE
from torch import optim
from modules.losses import Lossv2
from early_stopping import EarlyStopping
from tqdm import tqdm
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.preprocessing import scale
from scipy.stats import spearmanr
import umap.umap_ as umap
import math
from anndata import AnnData
warnings.filterwarnings('ignore')
import pandas as pd
import os
from sklearn.decomposition import PCA
from plotting import plot_latent

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

tissue = 'Tonsil'
adata = sc.read_visium(path=f'/ix/hosmanbeyoglu/kor11/CytAssist/{tissue}/GEX_PEX')
adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
adt_ref = pd.read_csv('/ix/hosmanbeyoglu/kor11/CytAssist/Brain/GEX_PEX/CytAssist_FFPE_Protein_Expression_Human_Glioblastoma_feature_reference.csv')
pdata = adata[:, [i in adt_ref[adt_ref.isotype_control==False].id.values for i in adata.var_names]]
adata = adata[:, [i not in adt_ref[adt_ref.isotype_control==False].id.values and 'MT-' not in i for i in adata.var_names]]
pdata.var.feature_types = 'Antibody Capture'


sc.pp.filter_genes(adata, min_counts=5)
sc.pp.filter_cells(adata, min_counts=5)
pdata.obsm['spatial'] = adata.obsm['spatial']
pdata.raw = pdata
pdata.X = pdata.X.astype(float)
data11 = adata.X.toarray()
data12 = pdata.X.toarray()

sc.tl.pca(pdata)
sc.pp.neighbors(pdata)
sc.tl.leiden(pdata, resolution=0.3)

type1 = pdata.obs.leiden.values
type2 = pdata.obs.leiden.values

gex = featurize(adata, pca_dim=1024)
pex = featurize(pdata, pca_dim=1024, clr=False)
d11, d12 = gex.features.cpu().numpy(), pex.features.cpu().numpy()
corr = torch.eye(d11.shape[0], d12.shape[0]).cuda()
pca1 = PCA(n_components=256).fit(d11)
pca2 = PCA(n_components=28).fit(d12)
d11_pca = pca1.transform(d11)
d12_pca = pca2.transform(d12)
d11 = torch.tensor(d11_pca).cuda()
d12 = torch.tensor(d12_pca).cuda()



import wandb

def main():
    wandb.init(save_code=True, dir='/ix/hosmanbeyoglu/kor11/tmp')
    
    lr = wandb.config['learning_rate']
    wd = wandb.config['weight_decay']
    
    
    model = SpatialVAE([d11.shape[1], d12.shape[1]], 32).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    es = EarlyStopping(patience=5000, verbose=False, delta=1e-4, path='gvae.pth')
    loss_func = Lossv2()
    
    wandb.watch(model)

    loss_func.alpha = {
        'kl_gex': wandb.config['kl_gex'],
        'kl_pex': wandb.config['kl_pex'],
        'recons_gex': wandb.config['recons_gex'],
        'recons_pex': wandb.config['recons_pex'],
        'cosine': wandb.config['cosine'],
        'consistency': wandb.config['consistency'],
        'adj': wandb.config['adj'],
        'spatial': wandb.config['spatial'],
        'alignment': wandb.config['alignment'],
    }
        
    epochs = 50000
    oracle = []
    feats = d11.data.cpu().numpy()
    feats2 = d12.data.cpu().numpy()
    

    with tqdm(total=epochs) as pbar:
        for e in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X=[d11, d12], A=gex.adj_norm, corr=corr)
            output.epochs = epochs
            output.gex_features_pca = d11
            output.adj_label = gex.adj_label
            output.pos_weight = gex.pos_weight
            output.gex_sp_dist = gex.sp_dists
            output.corr = gex.adj_label
            output.norm = gex.norm
            output.pex_features_pca = d12
            
            kl_loss_gex, kl_loss_pex, recons_loss_gex, recons_loss_pex, cosine_loss, consistency_loss, adj_loss, spatial_loss, alignment_loss = loss_func.compute(e, output)
            loss = kl_loss_gex+kl_loss_pex+recons_loss_gex+recons_loss_pex+cosine_loss+consistency_loss+adj_loss+spatial_loss+alignment_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            # losses.append(float(loss))
            
            es(-1*np.mean(oracle), model)   
            if es.early_stop: 
                model = es.best_model
                break
            
            with torch.no_grad():
                if e == 0 or e % 100 == 0:
                    model.eval()
                    
                    gex_recons = output.gex_recons.data.cpu().numpy()
                    gex_cor_mean = np.mean([spearmanr(feats[:, ixs], gex_recons[:, ixs]).statistic for ixs in range(gex_recons.shape[1])])

                    pex_recons = output.pex_recons.data.cpu().numpy()
                    pex_cor_mean = np.mean([spearmanr(feats2[:, ixs], pex_recons[:, ixs]).statistic for ixs in range(pex_recons.shape[1])])
                    
                
                        
                    proteins = model.decoders[1](model.fc_mus[0](model.encoders[0](d11, gex.adj_norm), gex.adj_norm))
                    corrsx = []
                    c = proteins.detach().cpu().numpy()
                    d = d12.cpu().numpy()
                    for ixs in range(d.shape[1]):
                        corrsx.append(spearmanr(d[:, ixs], c[:, ixs]).statistic)    
                        
                    oracle.append(np.mean(corrsx))                   
                    
                    wandb.log({
                        "gene_recons_corr": gex_cor_mean,
                        "protein_recons_corr": pex_cor_mean,
                        "oracle": np.mean(corrsx),
                        "total_loss": loss,
                        "aligment_loss": alignment_loss,
                        "cosine_loss": cosine_loss,
                        "spatial_loss": spatial_loss,
                        "adj_loss": adj_loss,
                        "kl_gex_loss": kl_loss_gex,
                        "kl_pex_loss": kl_loss_pex,
                        "consistency_loss": consistency_loss,
                        "MSE_Gex": recons_loss_gex,
                        "MSE_Pex": recons_loss_pex,
                        "epoch": e,
                    })
                    
                    if np.isnan(gex_cor_mean) or np.isnan(pex_cor_mean) or np.isnan(np.mean(corrsx)):
                        print("Breaking due to NaNs")
                        break   
                    

        
            pbar.update()
            pbar.set_description(f'Oracle: {np.mean(oracle):.3f} | GexCorr: {gex_cor_mean:.3f} | PexCorr: {pex_cor_mean:.3f}')



        model = es.best_model
        torch.save(model.state_dict(), wandb.run.name + '.pth')
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(wandb.run.name + '.pth')
        wandb.log_artifact(artifact)
        os.remove(wandb.run.name + '.pth')

        # model.eval()
        # integrated_data =  model(X=[d11, d12], A=gex.adj_norm, corr=corr)
        # gex_z = integrated_data.gex_z.data.cpu().numpy()
        # pex_z = integrated_data.pex_z.data.cpu().numpy()

        # a, b = plot_latent([gex_z, pex_z], [type1, type2], 
        #     ['Gene\nEmbedding', 'Protein\nEmbedding'], 
        #     legend=False, method='umap',
        #     separate_dim=False, save=f'{wandb.run.name}_umap.svg');
        
        # wandb.log({"umap": wandb.Image(f"{wandb.run.name}_umap.svg")})
        # os.remove(f'{wandb.run.name}_umap.svg')
        
        
        # plt.rcParams['figure.figsize'] = (8, 4)
        # output = integrated_data
        # a = output.gex_recons.data.cpu().numpy()
        # b = d11.data.cpu().numpy()
        # c = output.pex_recons.data.cpu().numpy()
        # d = d12.data.cpu().numpy()

        # modelx = SpatialVAE([d11.shape[1], d12.shape[1]], 32).cuda()
        # modelx.eval()
        # outputx =  modelx(X=[d11, d12], A=gex.adj_norm, corr=corr)
        # gex_z = integrated_data.gex_z.data.cpu().numpy()
        # pex_z = integrated_data.pex_z.data.cpu().numpy()

        # ax = outputx.gex_recons.data.cpu().numpy()
        # bx = d11.data.cpu().numpy()
        # cx = outputx.pex_recons.data.cpu().numpy()
        # dx = d12.data.cpu().numpy()

        # corrs = []
        # for ixs in range(b.shape[1]):
        #     corrs.append(spearmanr(b[:, ixs], a[:, ixs]).statistic)    
        # sns.histplot(corrs, color='red', alpha=0.5, label='TrainedModel')
        # corrs = []
        # for ixs in range(bx.shape[1]):
        #     corrs.append(spearmanr(bx[:, ixs], ax[:, ixs]).statistic + 0.15)    
        # sns.histplot(corrs, color='slateblue', alpha=0.5, label='ShuffledModel')
        # plt.title('Gene Expression Correlation')
        # plt.xlabel('Spearmanr')
        # plt.xlim(-0.35, 1.0)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f'{wandb.run.name}_gex_corr.svg', format='svg', dpi=200)
        # plt.close()

        # corrs = []
        # for ixs in range(d.shape[1]):
        #     corrs.append(spearmanr(d[:, ixs], c[:, ixs]).statistic)    
        # sns.histplot(corrs, color='green', alpha=0.5, label='TrainedModel')
        # corrs = []
        # for ixs in range(dx.shape[1]):
        #     corrs.append(spearmanr(dx[:, ixs], cx[:, ixs]).statistic + 0.15)    
        # sns.histplot(corrs, color='aquamarine', alpha=0.5, label='ShuffledModel')
        # plt.title('Protein Expression Correlation')
        # plt.xlabel('Spearmanr')
        # plt.xlim(-0.35, 1.0)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f'{wandb.run.name}_pex_corr.svg', format='svg', dpi=200)
        # plt.close()
        
        # os.remove(f'{wandb.run.name}_gex_corr.svg')
        # os.remove(f'{wandb.run.name}_pex_corr.svg')

    
if __name__ == '__main__':
    main()
