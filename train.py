import warnings
warnings.filterwarnings('ignore')

import uniport as up
from utils import featurize, train_test_split
from spicess.modules.losses import Metrics, Loss
from early_stopping import EarlyStopping
from spicess.vae_infomax import InfoMaxVAE

from plotting import plot_umap_grid, plot_latent
import wandb
import os
import scanpy as sc
from torch import optim
from tqdm import tqdm
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from scipy.stats import spearmanr
import squidpy as sq
import math
from anndata import AnnData
import pandas as pd
from sklearn.decomposition import PCA
import cuml
from matplotlib.colors import ListedColormap
from scipy.spatial import distance_matrix

import torch
assert torch.cuda.is_available()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

hex_colors =  ["#4a3e38",
"#9fbfa2",
"#b99bbd",
"#53366c",
"#b2834b",
"#00612e",
"#00acef",
"#00eee9",
"#ba4850",
"#cac84d",
"#ce52ad",
"#76d05c",
"#794bc8",
"#e85a00"]
colors = ListedColormap(sns.color_palette(hex_colors))
antibody_panel = pd.read_csv('./notebooks/antibody_panel.csv')

tissue = 'Breast'
with open(f'./notebooks/{tissue}.txt', 'r') as f:
        genes = [g.strip() for g in f.readlines()]
    
adata = sc.read_10x_h5(f'/ix/hosmanbeyoglu/kor11/CytAssist/{tissue}/GEX_PEX/filtered_feature_bc_matrix.h5', gex_only=False)
visium_ = sc.read_visium(path=f'/ix/hosmanbeyoglu/kor11/CytAssist/{tissue}/GEX_PEX/')
adata.uns['spatial'] = visium_.uns['spatial']
adata.obsm['spatial'] = visium_.obsm['spatial']
adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
adata.var["mt"] = adata.var_names.str.startswith("MT-")

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt


def train_test_split(adata):
        adata.obs['idx'] = list(range(0, len(adata)))
        xy = adata.obsm['spatial'].astype(float)
        x = xy[:, 0]
        y = xy[:, 1]
        xmin, ymin = adata.obsm['spatial'].min(0)
        xmax, ymax = adata.obsm['spatial'].max(0)
        x_segments = np.linspace(xmin, xmax, 4)
        y_segments = np.linspace(ymin, ymax, 4)
        category = np.zeros_like(x, dtype=int)
        for i, (xi, yi) in enumerate(zip(x, y)):
            for j, (xmin_j, xmax_j) in enumerate(zip(x_segments[:-1], x_segments[1:])):
                if xmin_j <= xi <= xmax_j:
                    for k, (ymin_k, ymax_k) in enumerate(zip(y_segments[:-1], y_segments[1:])):
                        if ymin_k <= yi <= ymax_k:
                            category[i] = 3*k + j
                            break
                    break

        adata.obs['train_test'] = category
        adata.obs['train_test'] = adata.obs['train_test'].astype('category')

train_test_split(adata)


sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
adata = adata[adata.obs["pct_counts_mt"] < 20]
adata = adata[:, adata.var["mt"] == False]
pdata = adata[:, adata.var.feature_types=='Antibody Capture']
adata = adata[:, adata.var.feature_types=='Gene Expression']
pdata.var["isotype_control"] = (pdata.var_names.str.startswith("mouse") | pdata.var_names.str.startswith("rat") | pdata.var_names.str.startswith("HLA"))
pdata = pdata[:, pdata.var.isotype_control==False]
adata.layers['counts'] = adata.X
pdata.layers['counts'] = pdata.X
adata.var_names_make_unique()
adata.shape, pdata.shape   

adata = adata[:, adata.var_names.isin(genes)]
adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
adata.obs['source'] = 'Breast CytAssist (Ref)'
adata.obs['domain_id'] = 0
adata.obs['source'] = adata.obs['source'].astype('category')
adata.obs['domain_id'] = adata.obs['domain_id'].astype('category')


sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
up.batch_scale(adata)

sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata)
sc.tl.leiden(adata)

adata_cyt = up.Run(
    name=tissue, 
    adatas=[adata], 
    adata_cm =adata, 
    lambda_s=1.0, 
    out='project', 
    ref_id=1, 
    outdir='./notebooks/output'
)
adata_cyt.obsm['latent'] = adata_cyt.obsm['project']

pdata.obsm['spatial'] = adata.obsm['spatial']
pdata.raw = pdata
adata.raw = adata
pdata.X = pdata.X.astype(float)

gex = featurize(adata)
pex = featurize(pdata, clr=True)


d11 = adata_cyt.obsm['latent']
d12 = pex.features.cpu().numpy()
A = gex.adj_norm.to_dense()
refLabels = adata_cyt.obs.leiden.values

tensify = lambda x: torch.tensor(x).float().cuda()

class Dojo:
    def __init__(self, gene_mtx, protein_mtx, adata, pdata, params):
        self.gene_mtx = gene_mtx
        self.protein_mtx = protein_mtx
        self.adata = adata
        self.pdata = pdata
        self.params = params
        
    def get_next_indices(self):
        for ix in self.adata.obs['train_test'].unique():
            mask = self.adata.obs['train_test'].isin([ix])
            train_idx = self.adata[~mask].obs.idx.values
            test_idx = self.adata[mask].obs.idx.values
            yield train_idx, test_idx
       
    def get_input_pairs(self, train_idx, test_idx):
        d12t = tensify(self.protein_mtx[test_idx, :])
        d11t = tensify(self.gene_mtx[test_idx, :])
        d12 = tensify(self.protein_mtx[train_idx, :])
        d11 = tensify(self.gene_mtx[train_idx, :])
        
        return d11, d12, d11t, d12t

            
    def train_model(self, train_idx, test_idx, idx):
        params = self.params
        
        d11, d12, d11t, d12t = self.get_input_pairs(train_idx, test_idx)
        
        model = InfoMaxVAE([d11.shape[1], d12.shape[1]], latent_dim=params['latent_dim']).cuda()
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
        es = EarlyStopping(model, patience=100, verbose=False, delta=0)
        metrics = Metrics(track=False)
        
        epochs = params['epochs']
        
        loss_func = Loss(max_epochs=epochs)
        loss_func.alpha = {
            'kl_gex': params['kl_gex'],
            'kl_pex': params['kl_pex'],
            'recons_gex': params['recons_gex'],
            'recons_pex': params['recons_pex'],
            'cosine_gex': params['cosine_gex'],
            'cosine_pex': params['cosine_pex'],
            'adj': params['adj'],
            'spatial': params['spatial'],
            'mutual_gex': params['mutual_gex'],
            'mutual_pex': params['mutual_pex']    
        }
        
        losses = []

        feats = d11t[:, :].data.cpu().numpy()
        feats2 = d12t[:, :].data.cpu().numpy()

        A = gex.adj_norm.to_dense()[train_idx, :][:, train_idx]
        At = gex.adj_norm.to_dense()[test_idx, :][:, test_idx]
        
        with tqdm(total=epochs) as pbar:
            for e in range(epochs):
                model.train()
                optimizer.zero_grad()

                output = model(X=[d11, d12], A=A)            

                output.epochs = epochs
                output.adj_label = gex.adj_label[train_idx][:, train_idx]
                output.pos_weight = gex.pos_weight
                output.gex_sp_dist = gex.sp_dists[train_idx][:, train_idx]
                output.corr = gex.adj_label[train_idx][:, train_idx]
                output.norm = gex.norm

                buffer = loss_func.compute(e, output)
                loss = metrics(buffer).sum()
                if idx == -100: wandb.log({"loss": loss}, step=e)

                
                for name, value in buffer.__dict__.items(): 
                    if idx == -100: wandb.log({name: value}, step=e)
                
                            
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                losses.append(float(loss))

                model.eval()
                output = model(X=[d11t, d12t], A=At) 
                pex_recons = output.pex_recons.data.cpu().numpy()
                pex_cor_mean = np.mean([spearmanr(feats2[:, ixs], pex_recons[:, ixs]).statistic for ixs in range(pex_recons.shape[1])])
                metrics.update_value('pex_corr', pex_cor_mean)
                if idx == -100: wandb.log({'pex_corr': pex_cor_mean}, step=e)

                
                proteins = model.impute(d11t, At)
                oracle_corr = np.mean([spearmanr(feats2[:, ixs], proteins[:, ixs]).statistic for ixs in range(proteins.shape[1])])
                metrics.update_value('oracle', oracle_corr, track=True)
                if idx == -100: wandb.log({'imputed_corr': oracle_corr}, step=e)

                cosine_loss = 0.5*(metrics.means.cosine_loss_gex + metrics.means.cosine_loss_pex)

                es(1-oracle_corr, model)
                if es.early_stop: 
                    model = es.best_model
                    break
                
                pbar.update()        
                pbar.set_description(f'{idx+1}/{9} | Oracle: {metrics.means.oracle:.3f} | PEX: {np.mean(metrics.means.pex_corr):.3f} | Cosine: {cosine_loss:.3g} ')  
        
        model.eval()
        proteins = model.impute(d11t, At)        
        return proteins, model
    
    def compute_spearman(self, real, preds):
        assert real.shape == preds.shape
        return [spearmanr(real[:, ixs], preds[:, ixs]).statistic for ixs in range(preds.shape[1])]


    def run(self):
        full_idx = self.adata.obs.idx.values
        proteins_train, model = self.train_model(full_idx, full_idx, -100)
        
        stacked_imputations = []
        tests = []
        for idx, (train_idx, test_idx) in enumerate(self.get_next_indices()):
            p, _ = self.train_model(train_idx, test_idx, idx)
            stacked_imputations.append(p)
            tests.append(test_idx)
            
        tests = np.concatenate(tests)
        imps = np.concatenate(stacked_imputations)
        proteins_test = np.stack([i[1] for i in sorted(zip(tests, imps)) ])   
        
        oracle_train = self.compute_spearman(real=self.protein_mtx, preds=proteins_train)
        oracle_test = self.compute_spearman(real=self.protein_mtx, preds=proteins_test)
        
        return oracle_train, oracle_test, model
    
if __name__ == '__main__':
    
    wandb.init(save_code=True, dir='/tmp')
    params = wandb.config
    

    dojo = Dojo(gene_mtx=d11, protein_mtx=d12, adata=adata, pdata=pdata, params=params)    
    oracle_train, oracle_test, model = dojo.run()
    
    wandb.log({'oracle_train': np.mean(oracle_train)})
    wandb.log({'oracle_test': np.mean(oracle_test)})
    
    
    train_df = pd.DataFrame(oracle_train, index=pdata.var_names, columns=['CORR'])
    test_df = pd.DataFrame(oracle_test, index=pdata.var_names, columns=['CORR'])
    
    
    print(train_df)
    
    wandb.log({'train_df': wandb.Table(dataframe=train_df, allow_mixed_types=True)})
    wandb.log({'test_df': wandb.Table(dataframe=test_df, allow_mixed_types=True)})
    
    torch.save(model.state_dict(), wandb.run.name + '.pth')
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(wandb.run.name + '.pth')
    wandb.log_artifact(artifact)
    

    integrated_data =  model(X=[tensify(d11), tensify(d12)], A=A)
    gex_z = integrated_data.gex_z.data.cpu().numpy()
    pex_z = integrated_data.pex_z.data.cpu().numpy()

    
    emb, _ = plot_latent(
        [gex_z, pex_z], 
        [refLabels,  refLabels], 
        legend=True, method='umap',
        separate_dim=False, n_neighbors=300,
        save=f'{wandb.run.name}_latent.png', fmt='png');
    
    proteins = model.impute(tensify(d11), A)
    
    plot_umap_grid(emb, proteins, pdata.var_names, save=f'{wandb.run.name}_grid.png', fmt='png')

    wandb.log({"latent_space": wandb.Image(f'{wandb.run.name}_latent.png')})
    wandb.log({"umap_proteins": wandb.Image(f'{wandb.run.name}_grid.png')})
    
    os.remove(f'{wandb.run.name}_latent.png')
    os.remove(f'{wandb.run.name}_grid.png')
    os.remove(wandb.run.name + '.pth')
    
    
    