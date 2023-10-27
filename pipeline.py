from argparse import Namespace
from typing import Tuple
import pandas as pd
import scanpy as sc
import os
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import optim
from anndata import AnnData
from tqdm import tqdm
import uniport as up
from utils import clean_adata, clr_normalize_each_cell, featurize, graph_alpha, preprocess_graph, train_test_split
from spicess.modules.losses import Metrics, Loss
from early_stopping import EarlyStopping
from spicess.vae_infomax import InfoMaxVAE
import numpy as np
from scipy.stats import spearmanr
from datetime import datetime
import yaml
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from muon import prot as pt
import scipy

class CleanExit:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is KeyboardInterrupt:
            print('user interrupt')
            return True
        return exc_type is None
    
    


floatify = lambda x: torch.tensor(x).cuda().float()

UNIQUE_VAR_NAMES = [
    'CD163', 'CR2', 'PCNA', 'VIM', 'KRT5', 'CD68', 'CEACAM8', 'PTPRC_1',
    'PAX5', 'SDC1', 'PTPRC_2', 'CD8A', 'BCL2', 'CD19', 'PDCD1', 'ACTA2',
    'FCGR3A', 'ITGAX', 'CXCR5', 'EPCAM', 'MS4A1', 'CD3E', 'CD14', 'CD40',
    'PECAM1', 'CD4', 'ITGAM', 'CD27', 'CCR7', 'CD274'
    ]

VAR_NAMES = [
    'CD163', 'CR2', 'PCNA', 'VIM', 'KRT5', 'CD68', 'CEACAM8', 'PTPRC_1',
    'PAX5', 'SDC1', 'PTPRC_2', 'CD8A', 'BCL2', 'CD19', 'PDCD1', 'ACTA2',
    'FCGR3A', 'ITGAX', 'CXCR5', 'EPCAM', 'MS4A1', 'CD3E', 'CD14', 'CD40',
    'PECAM1', 'CD4', 'ITGAM', 'CD27', 'CCR7', 'CD274'
]

column_corr = lambda a, b: [spearmanr(a[:, ixs], b[:, ixs]).statistic for ixs in range(a.shape[1])]

class Pipeline:
    
    def __init__(self):
        pass
        
    def run(self):
        raise NotImplementedError
        

        
class IntegrateDatasetPipeline(Pipeline):
    def __init__(self, tissue: str):
        self.tissue = tissue
        print('Created data integration pipeline.')
        
        
    def run(self, adata, adata2) -> Tuple[AnnData, AnnData]:
        
        self.adata_cm = AnnData.concatenate(adata, adata2, join='inner')
        sc.pp.normalize_total(self.adata_cm)
        sc.pp.log1p(self.adata_cm)
        up.batch_scale(self.adata_cm)
        self.cat_a, self.cat_b = list(self.adata_cm.obs.domain_id.cat.categories)
        self.adata = self.adata_cm[self.adata_cm.obs.domain_id==self.cat_a]
        self.adata2 = self.adata_cm[self.adata_cm.obs.domain_id==self.cat_b]
        
        self.adata.obsm['spatial'] = adata.obsm['spatial']
        self.adata.uns['spatial'] = adata.uns['spatial']
        
        self.adata2.obsm['spatial'] = adata2.obsm['spatial']
        self.adata2.uns['spatial'] = adata2.uns['spatial']
        
        adata_cm = AnnData.concatenate(self.adata.copy(), self.adata2.copy(), join='inner')
        adata_cyt = up.Run(
            name=self.tissue, 
            adatas=[adata_cm], 
            adata_cm =adata_cm, 
            lambda_s=1.0, 
            out='project',
            outdir='/ihome/hosmanbeyoglu/kor11/tools/SPICESS/workshop/output',
            ref_id=1)
        
        adata_cyt.obsm['latent'] = adata_cyt.obsm['project']
        adata = adata_cyt[adata_cyt.obs.domain_id==self.cat_a]
        adata2 = adata_cyt[adata_cyt.obs.domain_id==self.cat_b]
        adata.obsm['spatial'] = self.adata.obsm['spatial']
        adata.uns['spatial'] = self.adata.uns['spatial']
        adata2.obsm['spatial'] = self.adata2.obsm['spatial']
        adata2.uns['spatial'] = self.adata2.uns['spatial']
        
        return adata, adata2
        
class LoadVisiumPipeline(Pipeline):
    def __init__(self, tissue: str, visium_dir: str, sample_id: int, name: str):
        super().__init__()
        self.tissue = tissue
        self.name = name
        self.visium_dir = visium_dir
        self.sample_id = sample_id
        
        print('Created 10x Visium data loader pipeline.')
        
        
    def run(self):
        adata3 = sc.read_visium(path=self.visium_dir)
        adata3.obsm['spatial'] = adata3.obsm['spatial'].astype(float)
        adata3.layers['counts'] = adata3.X
        adata3.var_names_make_unique()
        adata3.obs['source'] = self.name
        adata3.obs['domain_id'] = self.sample_id
        adata3.obs['source'] = adata3.obs['source'].astype('category')
        adata3.obs['domain_id'] = adata3.obs['domain_id'].astype('category')
        
        clean_adata(adata3)
        
        return adata3
                

class AbstractEvaluationPipeline(Pipeline):
    
    def __init__(self):
        pass
    
    
    
    
    
    
    
    
    
class LoadCytAssistPipeline(Pipeline):

    def __init__(self, 
            tissue: str, h5_file: str, geneset: str, 
            sample_id: int, name: str, 
            celltypes: list = None):
        super().__init__()
        self.tissue = tissue
        self.celltypes = celltypes
        self.h5_file = h5_file
        self.sample_id = sample_id
        self.name = name
        with open(geneset, 'r') as f:
            self.geneset = [g.strip() for g in f.readlines()]
            
        print('Created CytAssist data loader pipeline.')
        
        
    def load_data(self, h5_file: str, ) -> Tuple[AnnData, AnnData]:
        adata = sc.read_10x_h5(h5_file, gex_only=False)
        visium_ = sc.read_visium(path=os.path.dirname(h5_file))
        adata.uns['spatial'] = visium_.uns['spatial']  
        adata.obsm['spatial'] = visium_.obsm['spatial']
        adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        adata = adata[adata.obs["pct_counts_mt"] < 20]
        adata = adata[:, adata.var["mt"] == False]
        pdata = adata[:, adata.var.feature_types=='Antibody Capture']
        adata = adata[:, adata.var.feature_types=='Gene Expression']
        pdata.var["isotype_control"] = (pdata.var_names.str.startswith("mouse") \
            | pdata.var_names.str.startswith("rat") \
            | pdata.var_names.str.startswith("HLA"))
        pdata = pdata[:, pdata.var.isotype_control==False]
        adata.layers['counts'] = adata.X
        pdata.layers['counts'] = pdata.X
        adata.var_names_make_unique()
        if self.geneset is not None:        
            adata = adata[:, adata.var_names.isin(self.geneset)]
        adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
        adata.obs['source'] = self.name
        adata.obs['domain_id'] = self.sample_id
        adata.obs['source'] = adata.obs['source'].astype('category')
        adata.obs['domain_id'] = adata.obs['domain_id'].astype('category')
        if self.celltypes is not None:
            adata.obs['celltypes'] = self.celltypes   
            
        if 'PTPRC' in pdata.var_names:
            pdata.var_names = UNIQUE_VAR_NAMES
        pdata = pdata[:, pdata.var_names.isin(UNIQUE_VAR_NAMES)]
        
        clean_adata(adata)
        clean_adata(pdata)
        
        return adata, pdata
        
    def run(self):
        return self.load_data(self.h5_file)
        
    def __call__(self):
        return self.run()


class FeaturizePipeline(Pipeline):
    """
    `run()`:
        Preprocesses the input AnnData object and adds graph-related information to its `obsm` and `uns` attributes.
    
    Parameters
    ----------
    adata : AnnData
        The input AnnData object.
    min_max : bool, optional (default: True)
        Whether to apply min-max scaling to the input data.
    clr : bool, optional (default: False)
        Whether to apply center-log ratio (CLR) transformation to the input data.
    log : bool, optional (default: True)
        Whether to apply log2 transformation to the input data.
    resolution : float or None, optional (default: 0.3)
        The resolution parameter for the Leiden clustering algorithm. If None, clustering is not performed.
    layer_added : str, optional (default: 'normalized')
        The name of the layer to be added to the `layers` and `obsm` attributes of the AnnData object.
    
    Returns
    -------
    AnnData
        The preprocessed AnnData object with additional graph-related information in its `obsm` and `uns` attributes.
    """
    
    def __init__(self, neighbors=6, post_process=None, post_args=None):
        super().__init__()
        self.neighbors = neighbors
        self.post_process = post_process
        self.post_args = post_args
        print('Created featurization pipeline.')
        
        
    def make_graph(self, spatial_coords):
        adj = graph_alpha(spatial_coords, n_neighbors=self.neighbors)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = adj_label.toarray()
        
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        adj_norm = preprocess_graph(adj)
        coords = torch.tensor(spatial_coords).float()
        sp_dists = torch.cdist(coords, coords, p=2)
        
        return adj_label, adj_norm, pos_weight, norm, sp_dists
    
    def clr_normalize(self, X):
        """Take the logarithm of the surface protein count for each cell, 
        then center the data by subtracting the geometric mean of the counts across all cells."""
        def seurat_clr(x):
            s = np.sum(np.log1p(x[x > 0]))
            exp = np.exp(s / len(x))
            return np.log1p(x / exp)
        return np.apply_along_axis(seurat_clr, 1, X)

            
    def run(self, adata, min_max=True, clr=False, log=True, resolution=None, layer_used = None, layer_added='normalized') -> AnnData:
        
            
        if 'train_test' not in adata.obs:
            train_test_split(adata)
            
        if layer_used is None:
            features = csr_matrix(adata.X).toarray()
        else:
            features = adata.obsm[layer_used]
            
        if clr:
            features = pt.pp.clr(AnnData(features), inplace=False).X
            # features = self.clr_normalize(features)
            
        if log:
            features = np.log2(features+0.5)            
            
        if min_max:
            scaler = MinMaxScaler()
            features = np.transpose(scaler.fit_transform(np.transpose(features)))
            
            features = scaler.fit_transform(features)
            
            

        adata.obsm[layer_added] = np.array(features)
        
        
        # if resolution is not None:
        #     sc.pp.neighbors(adata, use_rep=layer_added)
        #     sc.tl.leiden(adata, resolution=resolution)
        
        adj_label, adj_norm, pos_weight, norm, sp_dists = self.make_graph(adata.obsm['spatial'])
        
        if 'leiden_colors' in adata.uns:
            adata.uns.pop('leiden_colors')
        
        adata.obsm['adj_label'] = adj_label
        adata.obsm['adj_norm'] = adj_norm
        adata.obsm['sp_dists'] = sp_dists.numpy()
        adata.uns['pos_weight'] = pos_weight
        adata.uns['norm'] = norm
        
        
        return adata
        
            



class InferencePipeline(Pipeline):
    
    def __init__(self, config_pth=None, model = None, tissue=None, proteins=None):
        super().__init__()
        
        if model is not None:
            self.model = model.cuda()
        else:
                
            with open(config_pth, 'r') as f:
                self.config = yaml.safe_load(f)
                
            self.model = InfoMaxVAE(
                [16, self.config['nproteins']], 
                latent_dim = self.config['latent_dim'], 
                dropout = self.config['dropout']
            ).cuda()
            
            self.tissue = self.config['tissue']
            self.proteins = self.config['proteins']
            
            self.model.load_state_dict(torch.load('../model_zoo/'+self.config['model']))
        
        self.model.eval()
        
        self.featurizer = FeaturizePipeline()
        
        if tissue is not None:
            self.tissue = tissue
        if proteins is not None:
            self.proteins = proteins
        print('Created inference pipeline.')
        
    def ingest(self, adata):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d13 = floatify(adata.obsm['latent'])
        adj = graph_alpha(adata.obsm['spatial'], n_neighbors=6)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = torch.tensor(adj_label.toarray()).to(device)
        adj_norm = preprocess_graph(adj).to(device)
        A2 = adj_norm.to_dense()
        
        return d13, A2

    def run(self, adata: AnnData, normalize: bool = False):
        assert isinstance(adata, AnnData), 'adata must be an AnnData object.'
        if 'adj_norm' in adata.obsm:
            del adata.obsm['adj_norm']
        adata = adata.copy()
        self.featurizer.run(adata, clr=False, min_max=True, log=False, layer_used='latent', resolution=None)
        adatax = adata
        scaler = MinMaxScaler()

        d13, A2 = self.ingest(adatax)
        imputed_proteins, z_latent = self.model.impute(d13, A2, return_z=True)

        if normalize:
            proteins_norm = pd.DataFrame(scaler.fit_transform(imputed_proteins), 
                    columns=self.proteins)
        else:
            proteins_norm = pd.DataFrame(imputed_proteins, 
                    columns=self.proteins)
        
        pdata_eval = AnnData(proteins_norm)
        
        ## clone metadata
        pdata_eval.obs = adatax.obs
        pdata_eval.obsm['spatial'] = adatax.obsm['spatial']
        pdata_eval.uns['spatial'] = adatax.uns['spatial']
        pdata_eval.obsm['embeddings'] = z_latent
        
        return pdata_eval
        
        
        



class TrainModelPipeline(Pipeline):
    """
    Pipeline for training a model using matched spatial gene expression and protein expression data.
    """
    
    def __init__(self, 
            tissue,
            adata: AnnData, 
            pdata: AnnData,
            adata_eval: AnnData ,
            pdata_eval: AnnData, 
            latent_dim: int = 16, 
            dropout: float = 0.0, 
            lr: float = 1e-3, 
            wd: float = 0.0,
            patience: int = 1000,
            delta: float = 1e-3,
            epochs: int = 10000,
            kl_gex: float = 1e-6, 
            kl_pex: float = 1e-6, 
            recons_gex: float = 1e-3, 
            recons_pex: float = 1e-3, 
            cosine_gex: float = 1e-3, 
            cosine_pex: float = 1e-3, 
            adj: float = 1e-6, 
            spatial: float = 1e-5, 
            mutual_gex: float = 1e-3, 
            mutual_pex: float = 1e-3,
            cross_validate: bool = True,
            save: bool = False):
        super().__init__()
        self.tissue = tissue
        self.adata = adata
        self.pdata = pdata
        self.epochs = int(epochs)
        self.adata_eval = adata_eval
        self.pdata_eval = pdata_eval
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.patience = patience
        self.delta = delta
        self.lr = lr
        self.wd = wd
        self.save = save
        self.featurizer = FeaturizePipeline()
        self.loss_func = Loss(max_epochs=epochs)

        self.loss_func.alpha = {
            'kl_gex': kl_gex,
            'kl_pex': kl_pex,
            'recons_gex': recons_gex,
            'recons_pex': recons_pex,
            'cosine_gex': cosine_gex,
            'cosine_pex': cosine_pex,
            'adj': adj,
            'spatial': spatial,
            'mutual_gex': mutual_gex,
            'mutual_pex': mutual_pex    
        }
        
        self.metrics = Metrics(track=True)
        self.cross_validate = cross_validate
        
        assert self.adata.shape[0] == self.pdata.shape[0], 'adata and pdata must have the same number of spots.'
        
        self.artifacts = Namespace(
            tissue=self.tissue,
            nspots = self.adata.shape[0],
            ngenes=self.adata.shape[1],
            nproteins=self.pdata.X.shape[1],
            epochs=self.epochs,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            patience=self.patience,
            delta=self.delta,
            lr=self.lr,
            wd=self.wd,
            kl_gex=self.loss_func.alpha['kl_gex'],
            kl_pex=self.loss_func.alpha['kl_pex'],
            recons_gex=self.loss_func.alpha['recons_gex'],
            recons_pex=self.loss_func.alpha['recons_pex'],
            cosine_gex=self.loss_func.alpha['cosine_gex'],
            cosine_pex=self.loss_func.alpha['cosine_pex'],
            adj=self.loss_func.alpha['adj'],
            spatial=self.loss_func.alpha['spatial'],
            mutual_gex=self.loss_func.alpha['mutual_gex'],
            mutual_pex=self.loss_func.alpha['mutual_pex']
        )
        
        print('Created training pipeline.')
    

    
    
    def subset_adata(self, adata, segment):
        """Return adata_train and adata_eval based on the segment"""
        a1 = adata[adata.obs.train_test!=segment, :]
        a2 = adata[adata.obs.train_test==segment, :]
        
        return a1, a2
    
    
    
    def run(self, show_for=None):
        train_test_split(self.adata)
        train_test_split(self.pdata)
        output = Namespace()
        if self.cross_validate:
            imputed_concat = np.ones((self.adata.shape[0], self.pdata.shape[1]))
            real_concat = np.ones((self.adata.shape[0], self.pdata.shape[1]))
            
            pbar = tqdm(total=len(self.adata.obs.train_test.unique()), desc='Cross-Validating')
                
            for segment in self.adata.obs.train_test.unique():
                a_train, a_eval = self.subset_adata(self.adata, segment=segment)
                p_train, p_eval = self.subset_adata(self.pdata, segment=segment)
                pbar.update()        
                out, _ = self.train(a_train, p_train, a_eval, p_eval, label=f'Patch: {segment}', show_for=show_for)
                imputed_concat[a_eval.obs.idx.values, :] = out.imputed_proteins
                real_concat[a_eval.obs.idx.values, :] = out.d14[:, :].data.cpu().numpy()
                
            corr = np.mean(column_corr(imputed_concat, real_concat))
            pbar.set_description(f'Cross-Validation: {corr:.3f}')
            pbar.close()
            
        with CleanExit():
            output, artifacts = self.train(self.adata, self.pdata, self.adata_eval, self.pdata_eval)
        # output, artifacts = self.train(self.adata_eval, self.pdata_eval, self.adata, self.pdata)
        
        
        ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
        name = f'{self.tissue}_{ts}'
        
        if self.save:
            torch.save(output.model.state_dict(), f'../model_zoo/model_{name}.pth')
            artifacts.model = f'model_{name}.pth'
            artifacts.proteins = list(self.pdata.var_names)
            with open(f'../model_zoo/config_{name}.yaml', 'w') as f:
                yaml.dump(vars(self.artifacts), f)
            pd.DataFrame(self.metrics.values.__dict__).to_csv(f'../model_zoo/kpi_{name}.csv', index=False)
            
            print(f'Saved model-config to ../model_zoo/config_{name}.yaml')
            
        return output
    
    

        
        
    def train(self, adata_train, pdata_train, adata_eval, pdata_eval, label='Training', show_for=None):
        ## 
        self.featurizer.run(adata_train, clr=False, min_max=True, log=True, layer_used='latent')
        self.featurizer.run(adata_eval, clr=False, min_max=True, log=True, layer_used='latent')
        
        self.featurizer.run(pdata_train, clr=True, min_max=True, log=False)
        self.featurizer.run(pdata_eval, clr=True, min_max=True, log=False)
        
        d11 = floatify(adata_train.obsm['latent'])
        d12 = floatify(pdata_train.obsm['normalized'])
        d13 = floatify(adata_eval.obsm['latent'])
        d14 = floatify(pdata_eval.obsm['normalized'])
        
        adj_label = floatify(pdata_train.obsm['adj_label'])
        pos_weight = floatify(pdata_train.uns['pos_weight'])
        sp_dists = floatify(pdata_train.obsm['sp_dists'])
        norm = floatify(pdata_train.uns['norm'])
        
        A = adata_train.obsm['adj_norm'].to_dense().cuda()
        A2 = adata_eval.obsm['adj_norm'].to_dense().cuda()
        
        model = InfoMaxVAE([d11.shape[1], d12.shape[1]], latent_dim=self.latent_dim, dropout=self.dropout).cuda()
        es = EarlyStopping(model, patience=self.patience, verbose=False, delta=self.delta)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)

        losses = []
        test_protein = d14[:, :].data.cpu().numpy()
        
        with tqdm(total=self.epochs, disable=label!='Training') as pbar:
            for e in range(self.epochs):

                model.train()
                optimizer.zero_grad()

                output = model(X=[d11, d12], A=A)            
                output.epochs = self.epochs
                output.adj_label = adj_label
                output.pos_weight = pos_weight
                output.gex_sp_dist = sp_dists
                output.norm = norm
                                
                buffer = self.loss_func.compute(e, output)
                loss = self.metrics(buffer).sum()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                losses.append(float(loss))
                model.eval()

                imputed_proteins = model.impute(d13, A2)
                
                oracle_corr = np.mean(column_corr(test_protein, imputed_proteins))
                self.metrics.update_value('oracle', oracle_corr, track=True)
                
                
                oracle_self = np.mean(column_corr(d12[:, :].data.cpu().numpy(), model.impute(d11, A)))
                
                self.metrics.update_value('oracle_self', oracle_self, track=True)
                
                
                es(1-self.metrics.means.oracle, model)
                if es.early_stop: 
                    model = es.best_model
                    break
                
                _imputation = self.metrics.means.oracle
                _self_imputation = self.metrics.means.oracle_self
                
                _loss = np.mean(losses)
                
                pbar.update()        
                pbar.set_description(
                    f'{label} > Imputation: {_imputation:.3f} | SelfImputation: {_self_imputation:.3f} | Loss: {_loss:.3g}'
                )  

        
        model = es.best_model
        model.eval()
        
        imputed_proteins = model.impute(d13, A2)
        
        
        
        output = Namespace()
        output.model = model
        output.imputed_proteins = imputed_proteins
        output.d11 = d11
        output.d12 = d12
        output.d13 = d13
        output.d14 = d14
        output.A = A
        output.A2 = A2
        output.metrics = self.metrics
        output.results = pd.DataFrame(column_corr(
            imputed_proteins, d14.detach().cpu().numpy()), columns=['CORR'], index=list(self.pdata.var_names))
        
        pbar.set_description(
            f'{label} > Imputation: {_imputation:.3f} | SelfImputation: {_self_imputation:.3f} | Loss: {_loss:.3g}'
        ) 
        pbar.close()
        
        
        return output, self.artifacts
