from argparse import Namespace
from collections import OrderedDict
from typing import Tuple
from matplotlib import pyplot as plt
import pandas as pd
import scanpy as sc
import os
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import optim
from anndata import AnnData
from tqdm import tqdm
from utils import ImageSlicer, clean_adata, graph_alpha, preprocess_graph, train_test_split
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
import glob
from sklearn.decomposition import PCA
from PIL import Image

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
        

        
# class IntegrateDatasetPipeline(Pipeline):
#     def __init__(self, tissue: str):
#         self.tissue = tissue
#         print('Created data integration pipeline.')
        
        
#     def run(self, adata, adata2) -> Tuple[AnnData, AnnData]:
        
#         self.adata_cm = AnnData.concatenate(adata, adata2, join='inner')
#         sc.pp.normalize_total(self.adata_cm)
#         sc.pp.log1p(self.adata_cm)
#         up.batch_scale(self.adata_cm)
#         self.cat_a, self.cat_b = list(self.adata_cm.obs.domain_id.cat.categories)
#         self.adata = self.adata_cm[self.adata_cm.obs.domain_id==self.cat_a]
#         self.adata2 = self.adata_cm[self.adata_cm.obs.domain_id==self.cat_b]
        
#         self.adata.obsm['spatial'] = adata.obsm['spatial']
#         self.adata.uns['spatial'] = adata.uns['spatial']
        
#         self.adata2.obsm['spatial'] = adata2.obsm['spatial']
#         self.adata2.uns['spatial'] = adata2.uns['spatial']
        
#         adata_cm = AnnData.concatenate(self.adata.copy(), self.adata2.copy(), join='inner')
#         adata_cyt = up.Run(
#             name=self.tissue, 
#             adatas=[adata_cm], 
#             adata_cm =adata_cm, 
#             lambda_s=1.0, 
#             out='project',
#             outdir='/ihome/hosmanbeyoglu/kor11/tools/SPICESS/workshop/output',
#             ref_id=1)
        
#         adata_cyt.obsm['latent'] = adata_cyt.obsm['project']
#         adata = adata_cyt[adata_cyt.obs.domain_id==self.cat_a]
#         adata2 = adata_cyt[adata_cyt.obs.domain_id==self.cat_b]
#         adata.obsm['spatial'] = self.adata.obsm['spatial']
#         adata.uns['spatial'] = self.adata.uns['spatial']
#         adata2.obsm['spatial'] = self.adata2.obsm['spatial']
#         adata2.uns['spatial'] = self.adata2.uns['spatial']
        
#         return adata, adata2

class LoadSeuratTonsilsPipeline(Pipeline):
    
    def __init__(self, path: str):
        super().__init__()
        meta = """
            c28w2r_7jne4i          0.170068
            esvq52_nluss5          0.173260
            exvyh1_66caqq          0.294406
            gcyl7c_cec61b          0.294406
            p7hv1g_tjgmyj          0.174317
            qvwc8t_2vsr67          0.294406
            tarwe1_xott6q          0.171821
            zrt7gl_lhyyar          0.294406""".split()

        self.metadata = dict(zip(
                [meta[i] for i in range(0, len(meta), 2)], 
                [meta[i+1] for i in range(0, len(meta)-1, 2)]
            )
        )
        
        self.path = path
        self.colData, self.rowData, self.spatial_coords, self.counts = self.load_data()
        print(f'Created Tonsils data loader pipeline with {len(self.metadata)} STs.')
        
    
    def load_data(self):
        path = self.path
        colData = pd.read_csv(path+'colData.csv', engine='pyarrow', index_col=0)  
        rowData = pd.read_csv(path+'rowData.csv', engine='pyarrow', index_col=0)
        spatial_coords = pd.read_csv(path+'spatial_coords.csv', engine='pyarrow', index_col=0)
        counts = pd.read_csv(path+'counts.csv', engine='pyarrow', index_col=0)
        counts = counts.T
        colData.index = counts.index
        spatial_coords.index = counts.index
        
        # ((16224, 27), (26846, 8), (16224, 2), (16224, 26846))
        assert colData.shape[0] == spatial_coords.shape[0]
        assert colData.shape[0] == counts.shape[0]    
        assert rowData.shape[0] == counts.shape[1]
        
        self.sample_ids = list(colData.sample_id.unique())
        
        return colData, rowData, spatial_coords, counts
        
    def build_adata(self, ix, reload_data=False):        
        if reload_data:
            self.colData, self.rowData, self.spatial_coords, self.counts = self.load_data()
            
        sample_id = self.sample_ids[ix]
        
        colData = self.colData
        rowData = self.rowData
        spatial_coords = self.spatial_coords
        counts = self.counts

        target_idx = colData[colData.sample_id == sample_id].index
        img = Image.open(self.path+f'{sample_id}.png')

        xy = spatial_coords.loc[target_idx].values 

        st_adata = AnnData(
            X=counts.loc[target_idx].astype(float),
            obs=colData.loc[target_idx],
            var=rowData
        )
        st_adata.uns = OrderedDict({'spatial': {sample_id: {'images': 
                            {'hires': np.array(img)}, 
                            'scalefactors': {'spot_diameter_fullres': 1.0, 
                            'tissue_hires_scalef': float(self.metadata[sample_id])}}}})
        st_adata.obsm['spatial'] = xy.astype(float)
        
        return st_adata
    
    def run(self, i):
        return self.build_adata(i)
        

class LoadH5ADPipeline(Pipeline):
    
    def load_h5(self, h5_file, ix):
        adata3 = sc.read_h5ad(h5_file)
        adata3.obs['source'] = f'{self.tissue} Sample {ix}'
        adata3.obs['domain_id'] = 100+int(ix)
        adata3.obs['source'] = adata3.obs['source'].astype('category')
        adata3.obs['domain_id'] = adata3.obs['domain_id'].astype('category')
        
        return adata3
        
    
    def __init__(self, tissue: str, h5_dir: str):
        super().__init__()
        self.tissue = tissue
        self.fnames = sorted(glob.glob(h5_dir+'/*.h5ad'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f'Created H5AD data loader pipeline with {len(self.fnames)} h5ad files.')
        
        
    def run(self, i) -> AnnData:
        _adata = self.load_h5(self.fnames[i], i)
        _adata.var_names = _adata.var.feature_name.values
        clean_adata(_adata)
        
        return _adata
        
        
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
class LoadCytAssistPipeline(Pipeline):

    def __init__(self, 
            tissue: str, h5_file: str, 
            sample_id: int, name: str,
            geneset: str = None, 
            celltypes: list = None):
        super().__init__()
        
        self.tissue = tissue
        self.celltypes = celltypes
        self.h5_file = h5_file
        self.sample_id = sample_id
        self.name = name
        if geneset is not None:
            with open(geneset, 'r') as f:
                self.geneset = [g.strip() for g in f.readlines()]
        else:
            self.geneset = None
            
        print('Created CytAssist data loader pipeline.')
        
        
    def load_data(self, h5_file: str, return_raw=False) -> Tuple[AnnData, AnnData]:
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
class BulkCytAssistLoaderPipeline(Pipeline):
    def __init__(self, tissue: str, data_dir: str, geneset: str = None):
        self.tissue = tissue
        self.data_dir = data_dir
        self.geneset = geneset
        self.fnames = sorted(os.listdir(data_dir))
        self.loaders = []
        for i, fname in enumerate(self.fnames):
            self.loaders.append(
                LoadCytAssistPipeline(
                    tissue=self.tissue, 
                    h5_file=data_dir+fname+'/outs/filtered_feature_bc_matrix.h5',
                    geneset=self.geneset,
                    sample_id = 200+int(i),
                    name = f'{self.tissue} Sample {i}',
                )   
            )
            
        print(f'Created Bulk CytAssist data loader pipeline with {len(self.fnames)} samples.')
    
    def run(self, i):
        return self.loaders[i].run()

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
    
    def __init__(self, clr, min_max, log, layer_added='normalized', neighbors=6):
        super().__init__()
        self.neighbors = neighbors
        self.clr = clr
        self.min_max = min_max
        self.log = log
        self.layer_added = layer_added
        
        
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

            
    def run(self, adata, clr=None, min_max=None, log=None, layer_used=None, layer_added=None) -> AnnData:
        
        # sc.pp.filter_cells(adata, min_genes=200)
        # sc.pp.filter_genes(adata, min_cells=3)
        
        # adata = adata.copy()
            
        if 'train_test' not in adata.obs:
            train_test_split(adata)
            

            
        if clr is None:
            clr = self.clr
        if log is None:
            log = self.log
        if min_max is None:
            min_max = self.min_max
        if layer_added is None:
            layer_added = self.layer_added
            
            
        sc.pp.normalize_total(adata, inplace=True)
        
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
        # print('Created inference pipeline.')
        
    def ingest(self, adata):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d13 = floatify(adata.obsm['latent'])
        adj = graph_alpha(adata.obsm['spatial'], n_neighbors=6)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = torch.tensor(adj_label.toarray()).to(device)
        adj_norm = preprocess_graph(adj).to(device)
        A2 = adj_norm.to_dense()
        
        return d13, A2

    # FIXME: This is a hacky way to get the imputed protein expression values.
    def run(self, adata: AnnData, normalize: bool = False):
        assert isinstance(adata, AnnData), 'adata must be an AnnData object.'
        if 'adj_norm' in adata.obsm:
            del adata.obsm['adj_norm']
        adata = adata.copy()
        self.featurizer.run(adata, clr=False, min_max=True, log=True, layer_used='latent', resolution=None)
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
            adata_test: AnnData = None,
            latent_dim: int = 64, 
            pca_dim: int = 256,
            dropout: float = 0.1, 
            lr: float = 2e-3, 
            wd: float = 0.0,
            patience: int = 300,
            delta: float = 1e-3,
            epochs: int = 5000,
            kl_gex: float = 1e-6, 
            kl_pex: float = 1e-6,
            kl_img: float = 1e-8, 
            recons_gex: float = 1e-3, 
            recons_pex: float = 1e-3, 
            recons_img: float = 1e-3,
            cosine_gex: float = 1e-4, 
            cosine_pex: float = 1e-4,
            align: float = 1e-4,
            cosine_img: float = 1e-4, 
            adj: float = 1e-6, 
            spatial: float = 1e-5, 
            mutual_gex: float = 1e-3, 
            mutual_pex: float = 1e-3,
            batch_size: int = 8,
            cross_validate: bool = True,
            use_histology: bool = False,
            pretrained: str = None,
            freeze_encoder: bool = True,
            save: bool = False):
        super().__init__()
        self.tissue = tissue
        self.adata = adata
        self.pdata = pdata
        self.adata_test = adata_test
        self.epochs = int(epochs)
        self.adata_eval = adata_eval
        self.pdata_eval = pdata_eval
        self.latent_dim = latent_dim
        self.pca_dim = pca_dim
        self.dropout = dropout
        self.patience = patience
        self.delta = delta
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.save = save
        self.use_histology = use_histology
        self.pretrained = pretrained
        self.freeze_encoder = freeze_encoder
        self.gene_featurizer = FeaturizePipeline(clr=False, min_max=True, log=True)
        self.protein_featurizer = FeaturizePipeline(clr=True, min_max=True, log=False)
        
        self.loss_func = Loss(max_epochs=epochs, use_hist=use_histology)

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
            'mutual_pex': mutual_pex,    
            'kl_img': kl_img,
            'recons_img': recons_img,
            'cosine_img': cosine_img,
            'align': align
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
    
    
    def plot_losses(self):
        
        colors = ["#b357c2",
        "#82b63c",
        "#6a69c9",
        "#5cb96c",
        "#d14485",
        "#4bbeb1",
        "#cf473f",
        "#387e4d",
        "#c77cb6",
        "#c8a94a",
        "#6b94d0",
        "#e38742",
        "#767d34",
        "#b9606c",
        "#a46737"]

        f, axs = plt.subplots(3, 4, figsize=(22, 10), dpi=160)

        ordered_losses = ['cosine_loss_gex', 'cosine_loss_pex', 'kl_loss_gex', 
                        'recons_loss_gex', 'mutual_info_loss', 'alignment_loss', 
                        'kl_loss_pex', 'recons_loss_pex', 'adj_loss', 'oracle',
                        'spatial_loss', 'oracle_self']

        for i, (loss_name, ax) in enumerate(zip(ordered_losses, axs.flatten())):
            ax.plot(self.metrics.values.__dict__[loss_name], color=colors[i])
            ax.set_title(loss_name, color=colors[i], fontsize=15, fontweight='bold')
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.tight_layout()
        plt.show()
            
    
    
    def run(self, show_for=None):
        output = Namespace()
        if self.cross_validate:
            train_test_split(self.adata)
            train_test_split(self.pdata)
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
            
        # with CleanExit():
        #     output, artifacts = self.train(self.adata, self.pdata, self.adata_eval, self.pdata_eval)
        # output, artifacts = self.train(self.adata_eval, self.pdata_eval, self.adata, self.pdata)
        
        output, artifacts = self.train(self.adata, self.pdata, self.adata_eval, self.pdata_eval)
        
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
    
    
    def train(self, adata_train, pdata_train, adata_eval, pdata_eval, label='Training'):
        
        self.gene_featurizer.run(adata_train)
        self.gene_featurizer.run(adata_eval)
        self.protein_featurizer.run(pdata_train)
        self.protein_featurizer.run(pdata_eval)
        
        # self.pca = PCA(self.pca_dim).fit(adata_train.obsm['normalized'])
        
        if self.pca_dim:
            adata_train.obsm['normalized'] = PCA(self.pca_dim).fit_transform(adata_train.obsm['normalized'])
            adata_eval.obsm['normalized'] = PCA(self.pca_dim).fit_transform(adata_eval.obsm['normalized'])
        
        # adata_train.obsm['normalized'] = self.pca.transform(adata_train.obsm['normalized'])
        # adata_eval.obsm['normalized'] = self.pca.transform(adata_eval.obsm['normalized'])
        
        d11 = floatify(adata_train.obsm['normalized'])
        d12 = floatify(pdata_train.obsm['normalized'])
        d13 = floatify(adata_eval.obsm['normalized'])
        d14 = floatify(pdata_eval.obsm['normalized'])
        
        
        adj_label = floatify(pdata_train.obsm['adj_label'])
        pos_weight = floatify(pdata_train.uns['pos_weight'])
        sp_dists = floatify(pdata_train.obsm['sp_dists'])
        norm = floatify(pdata_train.uns['norm'])
        
        A = adata_train.obsm['adj_norm'].to_dense().cuda()
        A2 = adata_eval.obsm['adj_norm'].to_dense().cuda()
        
        slicer_train = ImageSlicer(adata_train, size=64, grayscale=True)

        Y = floatify(np.stack([slicer_train(i) for i in range(len(slicer_train))])).transpose(1, -1).unsqueeze(1)
        
        print(d11.shape)
        
        model = InfoMaxVAE([d11.shape[1], d12.shape[1]], 
                    latent_dim = self.latent_dim,
                    use_hist = self.use_histology,
                    pretrained = self.pretrained,
                    freeze_encoder = self.freeze_encoder,
                    dropout = self.dropout).cuda()
        
        es = EarlyStopping(model, patience=self.patience, verbose=False, delta=self.delta)
        es.best_model = model
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)

        losses = []
        test_protein = d14[:, :].data.cpu().numpy()
                
        with tqdm(total=self.epochs, disable=label!='Training') as pbar:
            for e in range(self.epochs):
                
                model.train()
                if self.use_histology:
                    model.image_encoder.train()
                
                optimizer.zero_grad()
                                
                output = model(X=[d11, d12], Y=Y, A=A)            
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
                
                if self.use_histology: 
                    imputed_from_img = model.img2proteins(Y)
                    oracle_img = np.mean(column_corr(d12[:, :].data.cpu().numpy(), imputed_from_img))
                    self.metrics.update_value('oracle_img', oracle_img, track=True)
                
                es(1-self.metrics.means.oracle, model)
                if es.early_stop: 
                    model = es.best_model
                    break

                # TODO: Make pbar dynamic with show_for     
                desc_str = ""
                desc_str+=f"{label} > " 
                if self.use_histology:
                    desc_str+=f"FromH&E: {self.metrics.means.oracle_img:.3f} || "
                desc_str+=f"Imputation: {self.metrics.means.oracle:.3f} | "
                desc_str+=f"SelfImputation: {self.metrics.means.oracle_self:.3f} | "
                desc_str+=f"Loss: {np.mean(losses):.3g} | "
                desc_str+=f"Alignment: {self.metrics.means.alignment_loss:.3g}"
                
                pbar.update()        
                pbar.set_description(desc_str)  

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
        output.Y = Y
        output.A = A
        output.A2 = A2
        output.metrics = self.metrics
        output.results = pd.DataFrame(column_corr(
            imputed_proteins, d14.detach().cpu().numpy()), 
            columns=['CORR'], index=list(self.pdata.var_names))
        
        #TODO: Add results for img2proteins
        
        pbar.close()
        
        
        return output, self.artifacts
