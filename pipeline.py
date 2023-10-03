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
from utils import clean_adata, featurize, graph_alpha, preprocess_graph
from spicess.modules.losses import Metrics, Loss
from early_stopping import EarlyStopping
from spicess.vae_infomax import InfoMaxVAE
import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.sparse as sp

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
        
    def load(self):
        raise NotImplementedError
        
    def save(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
        
    def __call__(self):
        return self.run()

        
class IntegrateDatasetPipeline(Pipeline):
    def __init__(self, tissue, adata, adata2):
        self.tissue = tissue
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
        
        print('Created data integration pipeline.')
        
        
    def run(self) -> Tuple[AnnData, AnnData]:
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


class InferencePipeline(Pipeline):
    
    def __init__(self, config_pth: str):
        super().__init__()

        with open(config_pth, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.model = InfoMaxVAE(
            [16, self.config['nproteins']], 
            latent_dim = self.config['latent_dim'], 
            dropout = self.config['dropout']
        ).cuda()
        
        self.model.load_state_dict(torch.load('../model_zoo/'+self.config['model']))
        self.model.eval()
        
        self.tissue = self.config['tissue']
        
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

    def run(self, adata: AnnData, normalize: bool = True):
        assert isinstance(adata, AnnData), 'adata must be an AnnData object.'
        adatax = adata.copy()
        scaler = MinMaxScaler()

        d13, A2 = self.ingest(adatax)
        imputed_proteins, z_latent = self.model.impute(d13, A2, return_z=True)

        proteins_norm = pd.DataFrame(scaler.fit_transform(imputed_proteins), 
                columns=self.config['proteins'])
        
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

    adata (AnnData): 
        raw gene expression counts
    pdata (AnnData): 
        raw adt counts
    adata_eval (AnnData): 
        Annotated data matrix for gene expression evaluation. Defaults to None.
    pdata_eval (AnnData): 
        Annotated data matrix for protein expression evaluation. Defaults to None.
    latent_dim (int, optional): 
        Dimension of the latent space. Defaults to 16.
    dropout (float, optional): 
        Dropout rate. Defaults to 0.1.
    lr (float, optional): 
        Learning rate. Defaults to 2e-3.
    wd (float, optional): 
        Weight decay. Defaults to 0.0.
    patience (int, optional): 
        Number of epochs to wait for improvement before early stopping. Defaults to 200.
    delta (float, optional): 
        Minimum change in the monitored quantity to qualify as improvement. Defaults to 1e-3.
    epochs (int, optional): 
        Maximum number of epochs to train the model. Defaults to 5500.
    kl_gex (float, optional): 
        Weight for gene expression KL divergence loss. Defaults to 1e-6.
    kl_pex (float, optional): 
        Weight for protein expression KL divergence loss. Defaults to 1e-6.
    recons_gex (float, optional): 
        Weight for gene expression reconstruction loss. Defaults to 1e-3.
    recons_pex (float, optional): 
        Weight for protein expression reconstruction loss. Defaults to 1e-3.
    cosine_gex (float, optional): 
        Weight for gene expression cosine similarity loss. Defaults to 1e-3.
    cosine_pex (float, optional): 
        Weight for protein expression cosine similarity loss. Defaults to 1e-3.
    adj (float, optional): 
        Weight for adjacency matrix binary cross entropy loss. Defaults to 1e-6.
    spatial (float, optional): 
        Weight for spatial distance loss. Defaults to 1e-5.
    mutual_gex (float, optional): 
        Weight for gene expression mutual information loss. Defaults to 1e-3.
    mutual_pex (float, optional): 
        Weight for protein expression mutual information loss. Defaults to 1e-3.
        
    """
    
    def __init__(self, 
            tissue,
            adata: AnnData, 
            pdata: AnnData,
            adata_eval: AnnData ,
            pdata_eval: AnnData, 
            latent_dim: int = 16, 
            dropout: float = 0.1, 
            lr: float = 2e-3, 
            wd: float = 0.0,
            patience: int = 200,
            delta: float = 1e-3,
            epochs: int = 5500,
            kl_gex: float = 1e-6, 
            kl_pex: float = 1e-6, 
            recons_gex: float = 1e-3, 
            recons_pex: float = 1e-3, 
            cosine_gex: float = 1e-3, 
            cosine_pex: float = 1e-3, 
            adj: float = 1e-6, 
            spatial: float = 1e-5, 
            mutual_gex: float = 1e-3, 
            mutual_pex: float = 1e-3
        ):
        super().__init__()
        self.tissue = tissue
        self.adata = adata
        self.pdata = pdata
        self.epochs = epochs
        self.adata_eval = adata_eval
        self.pdata_eval = pdata_eval
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.patience = patience
        self.delta = delta
        self.lr = lr
        self.wd = wd
        
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
    
    

    def pre_process_inputs(self, adata, pdata, neighbors=6, layer='latent'):
        gex = featurize(adata, neighbors=neighbors)
        pex = featurize(pdata, neighbors=neighbors, clr=True)
        
        gex.features = adata.obsm[layer]
        
        return gex, pex
        
    def transfer_accuracy(self, z_genes, z_proteins, labels, n_neighbors=12):
        knn =  KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance').fit(z_genes, labels)
        return accuracy_score(knn.predict(z_proteins), labels)
        
    def run(self):
        gex, pex = self.pre_process_inputs(self.adata, self.pdata)
        gex_eval, pex_eval = self.pre_process_inputs(self.adata_eval, self.pdata_eval)
        refLabels = self.adata.obs.celltypes.values
        d11 = floatify(gex.features)
        d12 = floatify(pex.features)
        d13 = floatify(gex_eval.features)
        d14 = floatify(pex_eval.features)
        
        model = InfoMaxVAE([d11.shape[1], d12.shape[1]], latent_dim=self.latent_dim, dropout=self.dropout).cuda()
        es = EarlyStopping(model, patience=self.patience, verbose=False, delta=self.delta)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=80, verbose=False)

        
        test_acc = 0
        losses = []
        test_protein = d14[:, :].data.cpu().numpy()

        A = gex.adj_norm.to_dense()
        A2 = gex_eval.adj_norm.to_dense()

        with tqdm(total=self.epochs) as pbar:
            for e in range(self.epochs):

                model.train()
                optimizer.zero_grad()

                output = model(X=[d11, d12], A=A)            
                
                output.epochs = self.epochs
                output.adj_label = gex.adj_label
                output.pos_weight = gex.pos_weight
                output.gex_sp_dist = gex.sp_dists
                output.norm = gex.norm

                buffer = self.loss_func.compute(e, output)
                loss = self.metrics(buffer).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                losses.append(float(loss))
                
                model.eval()
                    
                evalo = model(X=[d11, d12], A=A) 
                imputed_proteins = model.impute(d13, A2)
                oracle_corr = np.mean(column_corr(test_protein, imputed_proteins))
                self.metrics.update_value('oracle', oracle_corr, track=True)
                test_acc = self.transfer_accuracy(
                    evalo.gex_z.detach().cpu().numpy(), 
                    evalo.pex_z.detach().cpu().numpy(), 
                    refLabels)
                self.metrics.update_value('test_acc', test_acc, track=True)      
                
                es(1-self.metrics.means.oracle, model)
                if es.early_stop: 
                    model = es.best_model
                    break
                
                # scheduler.step(1-self.metrics.means.oracle)
                
                _alignment = self.metrics.means.test_acc
                _imputation = self.metrics.means.oracle
                _loss = np.mean(losses)
                lr = scheduler.optimizer.param_groups[0]['lr']
                
                pbar.update()        
                pbar.set_description(
                    f'Imputation: {_alignment:.3f} || Alignment: {_imputation:.3f}% | Loss: {_loss:.3g} | lr: {lr:.1e}'
                )  
                pbar.set_postfix({'es-counter': es.counter+1})
                

                

        model = es.best_model
        model.eval()
        ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
        name = f'{self.tissue}_{ts}'
        torch.save(model.state_dict(), f'../model_zoo/model_{name}.pth')
        self.artifacts.model = f'model_{name}.pth'
        self.artifacts.proteins = list(self.pdata.var_names)
        with open(f'../model_zoo/config_{name}.yaml', 'w') as f:
            yaml.dump(vars(self.artifacts), f)
        pd.DataFrame(self.metrics.values.__dict__).to_csv(f'../model_zoo/kpi_{name}.csv', index=False)
        
        print(f'Saved model-config to ../model_zoo/config_{name}.yaml')
    
        output = Namespace()
        output.model = model
        output.d11 = d11
        output.d12 = d12
        output.d13 = d13
        output.d14 = d14
        output.A = A
        output.A2 = A2
        output.metrics = self.metrics
        

        return output
