import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scanpy as sc
import pandas as pd
import math 
import matplotlib.pyplot as plt
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm
from anndata import AnnData

import squidpy as sq
from scipy.sparse import csr_matrix
from anndata.utils import logger as ad_logger
ad_logger.disabled = True
## https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213264
overlaps = lambda items : set.intersection(*map(set, items))
datapath = Path('/ix/hosmanbeyoglu/kor11/SpaceCovid') ## TODO: soft code this

antibody_map = {'Armenian.Hamster.IgG.Isotype.Ctrl.CCTGTCATTAAGACT': 'IgG',
 'CAT.1..SLC7A1..AAGCTGCATTTCCTT': 'CAT1',
 'CCR3..CD193..TAGAACCGTATCCGT': 'CCR3',
 'CD102.GATATTCAGTGCGAC': 'CD102',
 'CD103.TTCATTAGCCCGCTG': 'CD104',
 'CD105.TATCCCTGCCTTGCA': 'CD105',
 'CD106.CGTTCCTACCTACCT': 'CD106',
 'CD107a..LAMP.1..AAATCTGTGCCGTAC': 'CD107a',
 'CD115..CSF.1R..TTCCGTTGTTGTGAG': 'CD115',
 'CD117..c.kit..TGCATGTCATCGGTG': 'CD117',
 'CD11a.AGAGTCTCCCTTTAG': 'CD11a',
 'CD11c.GTTATGGACGCTTGC': 'CD11c',
 'CD120b..TNF.R.Type.II.p75..GAAGCTGTATCCGAA': 'CD120b',
 'CD122..IL.2Rb..GGTATGCGACACTTA': 'CD122',
 'CD124..IL.4Ra..GAACCGTAGTATAAC': 'CD124',
 'CD127..IL.7Ra..GTGTGAGGCACTCTT': 'CD127',
 'CD134..OX.40..CTCACCTACCTATGG': 'CD134',
 'CD135.GTAGCAAGATTCAAG': 'CD135',
 'CD137.TCCCTGTATAGATGA': 'CD137',
 'CD138..Syndecan.1..GCGTTTGTATGTACT': 'CD138',
 'CD14.AACCAACAGTCACGT': 'CD14',
 'CD140a.GTCATTGCGGTCCTA': 'CD140a',
 'CD150..SLAM..CAACGCCTAGAAACC': 'CD150',
 'CD152.AGTGTTTGTCCTGGT': 'CD152',
 'CD155..PVR..TAGCTTGGGATTAAG': 'CD155',
 'CD157..BST.1..AGAGCAATTTCAGGG': 'CD157',
 'CD159a..NKG2AB6..GTGTTTGTGTTCCTG': 'CD159a',
 'CD16.32.TTCGATGCTGGAGCA': 'CD16',
 'CD160.GCGTATGTCAGTACC': 'CD160',
 'CD163.GAGCAAGATTAAGAC': 'CD163',
 'CD169.Siglec.1.ATTGACGACAGTCAT': 'CD169',
 'CD170..Siglec.F..TCAATCTCCGTCGCT': 'CD170',
 'CD172a..SIRPa..GATTCCCTTGTAGCA': 'CD172a',
 'CD178..FasL..GTCACGTAGTATCTT': 'CD178',
 'CD182..CXCR2..TTTCCTTGTAGAGCG': 'CD182',
 'CD183..CXCR3..GTTCACGCCGTGTTA': 'CD183',
 'CD185..CXCR5..ACGTAGTCACCTAGT': 'CD185',
 'CD186..CXCR6..TGTCAGGTTGTATTC': 'CD186',
 'CD19.ATCAGCCATGTCAGT': 'CD19',
 'CD192..CCR2..AGTGCGATCTGCAAC': 'CD192',
 'CD194..CCR4..TTCATGTGTTTGTGC': 'CD194',
 'CD195..CCR5..ACCAGTTGTCATTAC': 'CD195',
 'CD196..CCR6..CTCTCTGCATTCCTC': 'CD196',
 'CD197..CCR7..TTATTAACAGCCCAC': 'CD197',
 'CD198..CCR8..ATCTCCGTTGTGCGA': 'CD198',
 'CD199..CCR9..CCCTCTGGTATGGTT': 'CD199',
 'CD1d..CD1.1..Ly.38..CAACTTGGCCGAATC': 'CD1d',
 'CD2.TTGCCGTGTGTTTAA': 'CD2',
 'CD20.TCCACTCCCTGTATA': 'CD20',
 'CD200..OX2..TCAATTCCGGTAGTC': 'CD200',
 'CD200R..OX2R..ATTCTTTCCCTCTGT': 'CD200R',
 'CD200R3.ATCAACTTGGAGCAG': 'CD200R3',
 'CD201..EPCR..TATGATCTGCCCTTG': 'CD201',
 'CD202b..Tie.2..TGTTTGTAAGTTCCC': 'CD202b',
 'CD204..Scavenger.R1..AGCTAGACACGTTGT': 'CD204',
 'CD205..DEC.205..CATATTGGCCGTAGT': 'CD205',
 'CD21.CD35..CR2.CR1..GGATAATTTCGATCC': 'CD21',
 'CD215..IL.15Ra..TGTACGCATGTATGG': 'CD215',
 'CD22.AGGTCCTCTCTGGAT': 'CD22',
 'CD226..DNAM.1..ACGCAGTATTTCCGA': 'CD226',
 'CD23.TCTCTTGGAAGATGA': 'CD23',
 'CD24.TATATCTTTGCCGCA': 'CD24',
 'CD25.ACCATGAGACACAGT': 'CD25',
 'CD252..OX40.Ligand..TCTCAGAACAGCCCT': 'CD252',
 'CD253..TRAIL..CCCTTTCCGATTCAA': 'CD253',
 'CD26..DPP.4..ATGGCCTGTCATAAT': 'CD26',
 'CD270..HVEM..GATCCGTGTTGCCTA': 'CD270',
 'CD272..BTLA..TGACCCTATTGAGAA': 'CD272',
 'CD273..B7.DC..PD.L2..CACTCCTTGTAGTCA': 'CD273',
 'CD274..B7.H1..PD.L1..TCGATTCCACCAACT': 'CD274',
 'CD278..ICOS..ACTGCCATATCCCTA': 'CD278',
 'CD279..PD.1..GAAAGTCAAAGCACT': 'CD279',
 'CD28.ATTAAGAGCGTGTTG': 'CD28',
 'CD3.GTATGTCCGCTCGAT': 'CD3',
 'CD300LG..Nepmucin..CGGTCCGTATCATTT': 'CD300LG',
 'CD300c.CD300d.MAIR.II.GTGATCTAAGATGCG': 'CD300c',
 'CD301b.CTTGCCTTGCGATTT': 'CD301b',
 'CD304..Neuropilin.1..CCAGCTCATTCAACG': 'CD304',
 'CD309..VEGFR2..Flk.1..AGTTGTCCTGTACGA': 'CD309',
 'CD31.GCTGTAGTATCATGT': 'CD31',
 'CD314..NKG2D..GAGGCTTATCATTTC': 'CD314',
 'CD317..BST2.PDCA.1..TGTGGTAGCCCTTGT': 'CD317',
 'CD326..Ep.CAM..ACCCGCGTTAGTATG': 'CD326',
 'CD335..NKp46..CCCTTTCACCTCGAA': 'CD335',
 'CD339..Jagged.1..TAGTATGCTGGAGCG': 'CD339',
 'CD34.GATTCCTTTACGAGC': 'CD34',
 'CD357..GITR..GGCACTCTGTAACAT': 'CD357',
 'CD36.TTTGCCGCTACGACA': 'CD36',
 'CD365..Tim.1..ATGGGATTAACCGTC': 'CD365',
 'CD366..Tim.3..ATTGGCACTCAGATG': 'CD366',
 'CD370..CLEC9A.DNGR1..AACTCAGTTGTGCCG': 'CD370',
 'CD371..CLEC12A..GCGAGAAATCTGCAT': 'CD371',
 'CD38.CGTATCCGTCTCCTA': 'CD38',
 'CD39.GCGTATTTAACCCGT': 'CD39',
 'CD4.AACAAGACCCTTGAG': 'CD4',
 'CD40.ATTTGTATGCTGGAG': 'CD40',
 'CD41.ACTTGGATGGACACT': 'CD41',
 'CD43.TTGGAGGGTTGTGCT': 'CD43',
 'CD45.1.CCTATGGACTTGGAC': 'CD45_1',
 'CD45.2.CACCGTCATTCAACC': 'CD45_2',
 'CD45.TGGCTATGGAGCAGA': 'CD45',
 'CD48.AGAACCGCCGTAGTT': 'CD48',
 'CD49a.CCATTCATTTGTGGC': 'CD49a',
 'CD49b.CGCGTTAGTAGAGTC': 'CD49b',
 'CD49d.CGCTTGGACGCTTAA': 'CD49d',
 'CD5.CAGCTCAGTGTGTTG': 'CD5',
 'CD51.GGAGTCAGGGTATTA': 'CD51',
 'CD54.ATAACCGACACAGTG': 'CD54',
 'CD55..DAF..ATTGTTGTCAGACCA': 'CD55',
 'CD62L.TGGGCCTAAGTCATC': 'CD62L',
 'CD63.ATCCGACACGTATTA': 'CD63',
 'CD68.CTTTCTTTCACGGGA': 'CD68',
 'CD69.TTGTATTCCGCCATT': 'CD69',
 'CD71.ACCGACCAGTAGACA': 'CD71',
 'CD73.ACACTTAACGTCTGG': 'CD73',
 'CD79b..Igb..TAACTCAGTGCGAGT': 'CD79b',
 'CD80.GACCCGGTGTCATTT': 'CD80',
 'CD83.TCTCAGGCTTCCTAG': 'CD83',
 'CD85k..gp49.Receptor..ATGTCAACTCTGGGA': 'CD85k',
 'CD86.CTGGATTTGTGTATC': 'CD86',
 'CD8a.TACCCGTAATAGCGT': 'CD8a',
 'CD8b..Ly.3..TTCCCTCTATGGAGC': 'CD8b',
 'CD9.TAGCAGTCACTCCTA': 'CD9',
 'CD90.2.CCGATCAGCCGTTTA': 'CD90',
 'CD93..AA4.1..early.B.lineage..GGTATTTCCTGTGGT': 'CD93',
 'CD94.CACAGTTGTCCGTGT': 'CD94',
 'CD95..Fas..CACATCGTTTGTGTA': 'CD95',
 'CX3CR1.CACTCTCAGTCCTAT': 'CX3CR1',
 'CXCR4.GTCGTGGTGTTGTTC': 'CXCR4',
 'DLL1.AGACCTCCTTACGAT': 'DLL1',
 'DR3..TNFRSF25..GCTTGGGCAATTAAG': 'DR3',
 'ENPP1..PC1..CATTAACCGCCCTTA': 'ENPP1',
 'ESAM.TATAGTTTCCGCCGT': 'ESAM',
 'F4.80.TTAACTTCAGCCCGT': 'F480',
 'FceRIa.AGTCACCTCGAAGCT': 'FceR1a',
 'Folate.Receptor.b..FR.b..CTCAGATGCCCTTTA': 'FRb',
 'GITR.L.GTATTCCGCACCTAT': 'GITR',
 'H.2Kb.bound.to.SIINFEKL.CGTTTATGGGATGGG': 'H2kb',
 'I.A.I.E.GGTCACCAGTATGAT': 'IAIE',
 'IL.21.Receptor.GATTCCGACAGTAGA': 'IL21',
 'IL.33Ra..IL1RL1.ST2..GCGATGGAGCATGTT': 'IL33Ra',
 'IgD.TCATATCCGTTGTCC': 'IgD',
 'IgM.AGCTACGCATTCAAT': 'IgM',
 'JAML.GTTATGGTTCGTGTT': 'JAML',
 'Ly.49A.AATTCCGTCAGATGA': 'Ly49A',
 'Ly.6A.E..Sca.1..TTCCTTTCCTACGCA': 'Ly6AE',
 'Ly.6C.AAGTCGTGAGGCATG': 'Ly6C',
 'Ly.6G.ACATTGACGCAACTA': 'Ly6G',
 'Ly.6G.Ly.6C..Gr.1..TAGTGTATGGACACG': 'Ly6C',
 'Ly108.CGATTCTTTGCGAGT': 'Ly108',
 'Ly49D.TATATCCCTCAACGC': 'Ly49D',
 'Ly49G.CGTATCTGTCATTAG': 'Ly49G',
 'Ly49H.CCAGTAGGCTTATTA': 'Ly49H',
 'MAdCAM.1.TTGGGCGATTAAGAA': 'MAdCAM',
 'MERTK..Mer..AGTAGAGCAACTCGT': 'MERTK',
 'Mouse.IgG1..k.isotype.Ctrl.GCCGGACGACATTAA': 'MouseIgG1',
 'Mouse.IgG2a...isotype.Ctrl.CTCCTACCTAAACTG': 'MouseIgG2a',
 'Mouse.IgG2b..k.isotype.Ctrl.ATATGTATCACGCGA': 'MouseIgG2b',
 'NK.1.1.GTAACATTACTCGTC': 'NK1',
 'Notch.1.TCCGGTCACTCAGTA': 'Notch1',
 'Notch.4.GTACTTAACGTCATC': 'Notch4',
 'P2X7R.TGCTTCATTCATGTG': 'P2X7R',
 'PIR.A.B.TGTAGAGTCAGACCT': 'PirAB',
 'Panendothelial.Cell.Antigen.CGTCCTAGTCATTGG': 'MECA32',
 'Rat.IgG1..k.isotype.Ctrl.ATCAGATGCCCTCAT': 'RatIgG1',
 'Rat.IgG1..u.Isotype.Ctrl.GGGAGCGATTCAACT': 'RatIgG1u',
 'Rat.IgG2a..k.Isotype.Ctrl.AAGTCAGGTTCGTTT': 'RatIgG2a',
 'Rat.IgG2b..k.Isotype.Ctrl.GATTCTTGACGACCT': 'RatIgG2b',
 'Rat.IgG2c..k.Isotype.Ctrl.TCCAGGCTAGTCATT': 'IgG2c',
 'Siglec.H.CCGCACCTACATTAG': 'Siglec.H',
 'TCR.Vb5.1..5.2.CTCAACAGTATTCTG': 'TCRVb5',
 'TCR.Vb8.1.8.2.ACTATCCGTTGTGCT': 'TCRVb8',
 'TCR.Vr1.1.Cr4.TCGTTTAACCAGCCT': 'TCRVr1',
 'TCR.Vr2.AAGCTGCACCGTAAT': 'TCRVr2',
 'TCR.Vr3.TCGTGGTCCCTTTCT': 'TCRVr3',
 'TCR.b.chain.TCCTATGGGACTCAG': 'TCRb',
 'TCR.r.theta.AACCCAAATAGCTGA': 'TCRr',
 'TER.119.Erythroid.Cells.GCGCGTTTGTGCTAT': 'TER119',
 'TIGIT..Vstm3..GAAAGTCGCCAACAG': 'TIGIT',
 'TLR4..CD284..MD2.Complex.GCAGTTGTCCGATTC': 'TLR4',
 'Tim.4.TGCTGGAGGGTATTC': 'Tim4',
 'X4.1BB.Ligand..CD137L..CAGTTCAGTACGCAG': 'X41BB',
 'anti.P2RY12.TTGCTTATTTCCGCA': 'P2RY12',
 'anti.human.mouse.CD49f.TTCCGAGGATGATCT': 'CD49f',
 'anti.human.mouse.integrin.b7.TCCTTGGATGTACCG': 'integrinb7',
 'anti.mouse.human.CD11b.TGAAGGCTCATTTGT': 'CD11b',
 'anti.mouse.human.CD15..SSEA.1..GCTAGTTTGTGCTGC': 'CD15',
 'anti.mouse.human.CD207.CGATTTGTATTCCCT': 'CD207',
 'anti.mouse.human.CD44.TGGCTTCAGGTCCTA': 'CD44',
 'anti.mouse.human.CD45R.B220.CCTACACCTCATAAT': 'CD45R',
 'anti.mouse.human.KLRG1..MAFA..GTAGTAGGCTAGACC': 'KLRG1',
 'anti.mouse.human.Mac.2..Galectin.3..GATGCAATTAGCCGG': 'Mac2',
 'anti.mouse.rat.CD29.ACGCATTCCTTGTGT': 'CD29',
 'anti.mouse.rat.CD61.TTCTTTACCCGCCTG': 'CD61',
 'anti.mouse.rat.CD62P..P.selectin..TGTGTGCCGTAGACT': 'CD62P',
 'anti.mouse.rat.XCR1.TCCATTACCCACGTT': 'XCR1',
 'anti.mouse.rat.human.CD27.CAAGGTATGTCACTG': 'CD27',
 'anti.rat.CD90.mouse.CD90.1.AGTATGGGATGCAAT': 'CD90'}

from scipy.sparse import csr_matrix
from utils import train_test_split, featurize
import torch

class Tissue:

    def __init__(self, adata, pdata=None, tissue_type=None, organism=None):
        self.adata = adata
        self.pdata = pdata
        self.pdata.obsm['spatial'] = self.adata.obsm['spatial']
        self.tissue_type = tissue_type
        self.organism = organism
        train_test_split(self.adata)
        if self.pdata is not None:
            self.pdata.obs['train_test'] = self.adata.obs['train_test']
            self.pdata.obs.train_test = self.pdata.obs.train_test.astype('category')
        self.adata.obs.train_test = self.adata.obs.train_test.astype('category')
        if isinstance(self.adata.X, csr_matrix):
            self.adata.X = self.adata.X.toarray()
        if isinstance(self.pdata.X, csr_matrix):
            self.pdata.X = self.pdata.X.toarray()
        
        self.gex = featurize(adata)
        self.pex = featurize(pdata)

        

class TissueCollection:
    def __init__(self, datapath: str) -> None:
        self.datapath = Path(datapath)
        self.data = []

    def load_data(self) -> None:
        pass
    
    def __getitem__(self, i: int) -> Tissue:
        "Returns a copy of the data."
        return self.data[i].copy()
    


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
            protein.columns = list(map(lambda x: antibody_map[x], protein.columns))
            protein = AnnData(protein)
            protein.X = csr_matrix(protein.X)
            rna = AnnData(rna.set_index('X'))
            rna.uns['protein'] = protein
            adata = rna
            xy = np.array([(a, b) for a, b in pd.Series(rna.obs.index).str.split('x').apply(
                lambda x: (int(x[0]), int(x[1]))).values])
            adata.obsm['spatial'] = xy
            rna.uns['protein'].obsm['spatial'] = xy
            adata.X = sp.csr_matrix(adata.X)
            adata.layers['counts'] = adata.X
            adata.uns['spatial'] = {}
            adata.uns['spatial'][rna_tsv] = {}
            adata.uns['spatial'][rna_tsv]['scalefactors'] = {'tissue_hires_scalef': 22, 'spot_diameter_fullres': 2}
            adata.uns['spatial'][rna_tsv]['images'] = {}
            adata.uns['spatial'][rna_tsv]['images']['hires'] = plt.imread(f'{str(datapath)}/{tissue.replace("mouse", "")}.jpg')

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
        
        # self.common_proteins = sorted(list(zero_perc_df[(zero_perc_df.mean(1) < self.cutoff)].index))
        # self.common_proteins = list(filter(None, map(self.parse_adts, self.common_proteins)))
        new_data = []
        for adata in self.data:
            prot = adata.uns['protein'].copy()
            # prot = prot[:, [i in self.common_proteins for i in map(self.parse_adts, prot.var_names)]]
            prot = prot[:, [list(self.common_proteins).index(i) for i in self.common_proteins]]
            prot.var_names = self.common_proteins
            adata.uns['protein'] = prot
            adata = adata[:, [cg in self.common_genes for cg in adata.var_names]]
            new_data.append(adata)
            
        self.data = new_data
            
    
    def plot_data(self):
        num_adatas = len(self.data)
        num_rows = math.ceil(num_adatas/4)
        fig, axs = plt.subplots(num_rows, 4, figsize=(12, 4*num_rows))
        axs = axs.flatten()
        for i in range(num_adatas):
            sq.pl.spatial_scatter(self.data[i], ax=axs[i], frameon=False)
            axs[i].set_title(f"{self.tissues[i]}\n{self.data[i].shape[0]} x {self.data[i].shape[1]} x {self.data[i].uns['protein'].shape[1]}", fontsize=10)
        for i in range(num_adatas, len(axs)):
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()