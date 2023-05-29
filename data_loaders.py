import numpy as np
import scanpy as sc
import pandas as pd
import math 
import matplotlib.pyplot as plt
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm
from anndata import AnnData
import warnings
warnings.filterwarnings('ignore')
import squidpy as sq

overlaps = lambda items : set.intersection(*map(set, items))
datapath = Path('/ix/hosmanbeyoglu/kor11/SpaceCovid') ## TODO: soft code this
proteins = [
    'IgG', 'CAT1', 'CCR3', 'CD102', 'CD104', 'CD105', 'CD106', 'CD107a', 'CD115', 'CD117', 'CD11a', 'CD11c', 'CD120b',
    'CD122', 'CD124', 'CD127', 'CD134', 'CD135', 'CD137', 'CD138', 'CD14', 'CD140a', 'CD150', 'CD152', 'CD155',
    'CD157', 'CD159a', 'CD16', 'CD160', 'CD163', 'CD169', 'CD170', 'CD172a', 'CD178', 'CD182', 'CD183', 'CD185',
    'CD186', 'CD19', 'CD192', 'CD194', 'CD195', 'CD196', 'CD197', 'CD198', 'CD199', 'CD1d', 'CD2', 'CD20', 'CD200',
    'CD200R', 'CD200R3', 'CD201', 'CD202b', 'CD204', 'CD205', 'CD21', 'CD215', 'CD22', 'CD226', 'CD23', 'CD24',
    'CD25', 'CD252', 'CD253', 'CD26', 'CD270', 'CD272', 'CD273', 'CD274', 'CD278', 'CD279', 'CD28', 'CD3', 'CD300LG',
    'CD300c', 'CD301b', 'CD304', 'CD309', 'CD31', 'CD314', 'CD317', 'CD326', 'CD335', 'CD339', 'CD34', 'CD357',
    'CD36', 'CD365', 'CD366', 'CD370', 'CD371', 'CD38', 'CD39', 'CD4', 'CD40', 'CD41', 'CD43', 'CD45_1', 'CD45_2',
    'CD45', 'CD48', 'CD49a', 'CD49b', 'CD49d', 'CD5', 'CD51', 'CD54', 'CD55', 'CD62L', 'CD63', 'CD68', 'CD69',
    'CD71', 'CD73', 'CD79b', 'CD80', 'CD83', 'CD85k', 'CD86', 'CD8a', 'CD8b', 'CD9', 'CD90', 'CD93', 'CD94', 'CD95',
    'CX3CR1', 'CXCR4', 'DLL1', 'DR3', 'ENPP1', 'ESAM', 'F480', 'FceR1a', 'FRb', 'GITR', 'H2kb', 'IAIE', 'IL21', 'IL33Ra',
    'IgD', 'IgM', 'JAML', 'Ly49A', 'Ly6AE', 'Ly6C', 'Ly6G', 'Ly6C', 'Ly108', 'Ly49D', 'Ly49G', 'Ly49H', 'MAdCAM',
    'MERTK', 'MouseIgG1', 'MouseIgG2a', 'MouseIgG2b', 'NK1', 'Notch1', 'Notch4', 'P2X7R', 'PirAB', 'MECA32',
    'RatIgG1', 'RatIgG1u', 'RatIgG2a', 'RatIgG2b', 'IgG2c', 'Siglec.H', 'TCRVb5', 'TCRVb8', 'TCRVr1', 'TCRVr2',
    'TCRVr3', 'TCRb', 'TCRr', 'TER119', 'TIGIT', 'TLR4', 'Tim4', 'X41BB', 'P2RY12', 'CD49f', 'integrinb7', 'CD11b',
    'CD15', 'CD207', 'CD44', 'CD45R', 'KLRG1', 'Mac2', 'CD29', 'CD61', 'CD62P', 'XCR1', 'CD27', 'CD90'
]

class DataBlob:
    
    def __init__(self):
        self.tissues = ['mousecolon', 'mouseintestine', 'mousekidney', 'mousespleen']
        self.data = self.injest_data()        
        self.common_proteins = list(overlaps([x.uns['protein'].var_names for x in self.data]))
        self.common_genes = list(overlaps([x.var_names for x in self.data]))
        self.cutoff = 0.1
        self.align_tissues_()
        
    def __getitem__(self, i: int) -> AnnData:
        "Returns a copy of the data."
        return self.data[i].copy()
        
    def __len__(self) -> int:
        return len(self.data)

    
    # def train_test_split(self, adata: AnnData):
    #     adata.obs['idx'] = list(range(0, len(adata)))

    #     xy = adata.obsm['spatial']
    #     x = xy[:, 0]
    #     y = xy[:, 1]

    #     xmin, ymin = adata.obsm['spatial'].min(0)
    #     xmax, ymax = adata.obsm['spatial'].max(0)

    #     x_segments = np.linspace(xmin, xmax, 4)
    #     y_segments = np.linspace(ymin, ymax, 4)

    #     category = np.zeros_like(x, dtype=int)

    #     for i, (xi, yi) in enumerate(zip(x, y)):
    #         for j, (xmin_j, xmax_j) in enumerate(zip(x_segments[:-1], x_segments[1:])):
    #             if xmin_j <= xi <= xmax_j:
    #                 for k, (ymin_k, ymax_k) in enumerate(zip(y_segments[:-1], y_segments[1:])):
    #                     if ymin_k <= yi <= ymax_k:
    #                         category[i] = 3*k + j
    #                         break
    #                 break

    #     adata.obs['train_test'] = category
    
    
    # def filter_adata_(self, adata: AnnData, min_genes: int = 200, target_sum: int = 1e4):
    #     sc.pp.filter_cells(adata, min_genes=min_genes)
    #     sc.pp.normalize_total(adata, target_sum=target_sum)
    #     sc.pp.log1p(adata)                
    #     sc.tl.pca(adata)
    #     sc.pp.neighbors(adata)
    #     sc.tl.umap(adata)
    #     sc.tl.leiden(adata)
    
    
    def load_data(self) -> tuple:
        xls = pd.ExcelFile('/ihome/hosmanbeyoglu/kor11/tools/41587_2022_1536_MOESM3_ESM.xlsx')
        
        sheet_to_df_map = {}
        for sheet_name in xls.sheet_names:
            sheet_to_df_map[sheet_name] = xls.parse(sheet_name, index_col=0, header=1)

        def load_replicate(replicate_num: int) -> tuple:
            if replicate_num == 1: sid = '1255334_1278495'
            else: sid = '1255333_1278494'
                
            an = sc.read_10x_h5(f'/ix/hosmanbeyoglu/kor11/SPOTS/GSE198353_spleen_rep_{replicate_num}_filtered_feature_bc_matrix.h5', gex_only=False)
            an.var_names_make_unique()
            spots_adata = sc.read_visium(f'/ix/hosmanbeyoglu/kor11/SPOTS/spatial_rep_{replicate_num}', 
                        count_file=f'/ix/hosmanbeyoglu/kor11/SPOTS/GSE198353_spleen_rep_{replicate_num}_filtered_feature_bc_matrix.h5')
            spots_adata.var_names_make_unique()
            spots_adata.uns['spatial'][sid]['images']['lowres'] = np.expand_dims(spots_adata.uns['spatial'][sid]['images']['lowres'], 2)
            spots_adata.uns['spatial'][sid]['images']['hires'] = np.expand_dims(spots_adata.uns['spatial'][sid]['images']['hires'], 2)
            common_idx = list(set(sheet_to_df_map['Spleen ADT Deconvolution'].query(f'Replicate == "Replicate {replicate_num}"').index).intersection(set(an.obs.index)))
            an = an[common_idx, :].copy()
            an.obs = sheet_to_df_map['Spleen ADT Deconvolution'].query(f'Replicate == "Replicate {replicate_num}"').join(an.obs, how='inner').drop('Replicate', axis=1)
            an.obs = an.obs.join(sheet_to_df_map['Spleen Metadata'].query(f'Replicate == "Replicate {replicate_num}"'), how='inner')
            protein = an[:, an.var["feature_types"] == "Antibody Capture"].copy()
            rna = an[:, an.var["feature_types"] == "Gene Expression"].copy()
            rna.uns = spots_adata[rna.obs.index, rna.var_names].uns.copy()
            rna.obsm = spots_adata[rna.obs.index, rna.var_names].obsm.copy()
            protein.uns = spots_adata[protein.obs.index, rna.var_names].uns.copy()
            protein.obsm = spots_adata[protein.obs.index, rna.var_names].obsm.copy()
            rna.obsm["protein_raw_counts"] = protein.X.copy()
            return protein, rna

        protein1, rna1 = load_replicate(1)
        protein2, rna2 = load_replicate(2)
        
        rna1.uns['protein'] = protein1
        rna2.uns['protein'] = protein2

        return rna1, rna2
        
        
    def injest_data(self) -> list[AnnData]:
        data_glob = []
    
        for tissue in tqdm(self.tissues, desc='Injesting Data...'):
            rna_tsv, protein_tsv = sorted([i.name for i in datapath.glob(f'*{tissue}*.tsv')], 
                                          key=lambda x: len(x.split('_')[-1].split('.')[0]))

            rna = pd.read_csv(datapath/rna_tsv, sep='\t', engine="pyarrow")
            protein = pd.read_csv(datapath/protein_tsv, sep='\t', engine="pyarrow")
            protein = protein.set_index('X').drop('unmapped', axis=1).loc[rna.X]
            protein = AnnData(protein)
            rna = AnnData(rna.set_index('X'))
            rna.uns['protein'] = protein
            adata = rna
            xy = np.array([(a, b) for a, b in pd.Series(rna.obs.index).str.split('x').apply(
                lambda x: (int(x[0]), int(x[1]))).values])
            adata.obsm['spatial'] = xy
            adata.X = sp.csr_matrix(adata.X)
            adata.layers['counts'] = adata.X
            adata.uns['spatial'] = {}
            adata.uns['spatial'][rna_tsv] = {}
            adata.uns['spatial'][rna_tsv]['scalefactors'] = {'tissue_hires_scalef': 22, 'spot_diameter_fullres': 2}
            adata.uns['spatial'][rna_tsv]['images'] = {}
            adata.uns['spatial'][rna_tsv]['images']['hires'] = plt.imread(f'{tissue}.jpg')

            data_glob.append(adata)
            
        return data_glob
    
    
    def align_tissues_(self):
        dfs = [self.data[i].uns['protein'].to_df() for i in range(len(self.data))]
        zero_perc = []
        for df in dfs:
            zero_perc.append((df == 0).sum(axis=0) / df.shape[0])

        zero_perc_df = pd.concat(zero_perc, axis=1)
        zero_perc_df.columns = [i[5:] for i in self.tissues]
        self.zero_perc_df = zero_perc_df
        self.common_proteins = sorted(list(zero_perc_df[(zero_perc_df.mean(1) < self.cutoff)].index))
        self.common_proteins = list(filter(None, map(self.parse_adts, self.common_proteins)))
        new_data = []
        for adata in self.data:
            prot = adata.uns['protein'].copy()
            prot = prot[:, [i in self.common_proteins for i in map(self.parse_adts, prot.var_names)]]
            # prot = prot[:, self.common_proteins]
            prot.var_names = self.common_proteins
            adata.uns['protein'] = prot
            adata = adata[:, [cg in self.common_genes for cg in adata.var_names]]
            self.train_test_split(adata)
            new_data.append(adata)
            
        self.data = new_data
            
    
    def plot_data(self):
        num_adatas = len(self.data)
        num_rows = math.ceil(num_adatas/4)
        fig, axs = plt.subplots(num_rows, 4, figsize=(12, 4*num_rows))
        axs = axs.flatten()
        for i in range(num_adatas):
            sq.pl.spatial_scatter(self.data[i], ax=axs[i], frameon=False)
            axs[i].set_title(f"{self.tissues[i]}")
        for i in range(num_adatas, len(axs)):
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()