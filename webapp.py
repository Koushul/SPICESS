import streamlit as st
import sys
import warnings

from plotting import plot_latent, plot_umap_grid
warnings.filterwarnings('ignore')
sys.path.append('.')
import pandas as pd
import uniport as up
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as cuml
import matplotlib.pyplot as plt
import seaborn as sns
import time
from modules.vae_infomax import InfoMaxVAE
from utils import featurize
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

UMAP = cuml.UMAP

st.set_page_config(layout="wide", page_icon='🌶️', page_title='SPICESS Web Portal')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Welcome to SPICESS!')
st.image('model.png', use_column_width=True)


antibody_panel = pd.read_csv('./notebooks/antibody_panel.csv')

with st.expander('Antibody Panel'):
    st.table(antibody_panel)
    
@st.cache_data
def load_data(tissue):
    adata = sc.read_10x_h5(f'/ix/hosmanbeyoglu/kor11/CytAssist/{tissue}/GEX_PEX/filtered_feature_bc_matrix.h5', gex_only=False)
    visium_ = sc.read_visium(path=f'/ix/hosmanbeyoglu/kor11/CytAssist/{tissue}/GEX_PEX/')
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
    
    return adata, pdata
        
# @st.cache_data
def project(ix, adata_ref, tissue):
    adata3 = sc.read_h5ad(f'/ix/hosmanbeyoglu/kor11/CytAssist/Breast/visium/patient_{ix}.h5ad')
    adata3.obs['source'] = f'Breast Sample {ix}'
    adata3.obs['domain_id'] = ix
    adata3.obs['source'] = adata3.obs['source'].astype('category')
    adata3.obs['domain_id'] = adata3.obs['domain_id'].astype('category')
    
    sc.pp.normalize_total(adata3)
    sc.pp.log1p(adata3)
    up.batch_scale(adata3)
    
    adata3.var_names = adata3.var.feature_name.values
    
    st.success(f'Sample {ix} has {adata3.shape[0]} spots and {adata3.shape[1]} genes.')
    
    adata_cm = AnnData.concatenate(adata3, adata_ref, join='inner')
    adata_new = up.Run(
        name=tissue, 
        adatas=[adata3, adata_ref], 
        adata_cm =adata_cm, 
        lambda_s=1.0, 
        out='project', 
        ref_id=1, 
        outdir='./notebooks/output'
    )
    adata3.obsm['latent'] = adata_new[adata_new.obs.domain_id==ix].obsm['project']
    adata3.obsm['project'] = adata_new[adata_new.obs.domain_id==ix].obsm['project']
    return adata3

@st.cache_data
def build_scaffold(tissue):
    
    with open(f'./notebooks/{tissue}.txt', 'r') as f:
        genes = [g.strip() for g in f.readlines()]
    
    adata, pdata = load_data(tissue)
    
    adata = adata[:, adata.var_names.isin(genes)]
    
    # adata6 = sc.read_h5ad('/ix/hosmanbeyoglu/kor11/CytAssist/Breast/visium/patient_6.h5ad')
    # adata.var_names_make_unique()
    # adata6.var_names_make_unique()
    
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
    # adata6.obsm['spatial'] = adata6.obsm['spatial'].astype(float)
    adata.obs['source'] = 'Breast CytAssist (Ref)'
    adata.obs['domain_id'] = 0
    adata.obs['source'] = adata.obs['source'].astype('category')
    adata.obs['domain_id'] = adata.obs['domain_id'].astype('category')
    
    # adata6.obs['source'] = 'Breast Sample 6'
    # adata6.obs['domain_id'] = 6
    # adata6.obs['source'] = adata6.obs['source'].astype('category')
    # adata6.obs['domain_id'] = adata6.obs['domain_id'].astype('category')
    # adata6.var_names = adata6.var.feature_name.values
    # adata_cm = AnnData.concatenate(adata, adata6, join='inner')
    # sc.pp.normalize_total(adata_cm)
    # sc.pp.log1p(adata_cm)
    # sc.pp.highly_variable_genes(adata_cm, n_top_genes=5000, inplace=False, subset=True)
    # up.batch_scale(adata_cm)
    # adata = adata_cm[adata_cm.obs.domain_id==0]
    # adata6 = adata_cm[adata_cm.obs.domain_id==6]
    
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    
    
    return adata, pdata



cols = st.columns(3)
# tissue = cols[0].selectbox('Select tissue (3)', ['Breast', 'Tonsil', 'Brain'])
tissue = st.radio('Load pre-trained model for which tissue?', ['Breast', 'Tonsil', 'Brain'], horizontal=True)


def clean_adata(adata):
    for v in ['mt', 'n_cells_by_counts', 'mean_counts', 'log1p_mean_counts', 'pct_dropout_by_counts', 'total_counts', 'log1p_total_counts', 
        'gene_ids', 'feature_types', 'genome', 'mt-0', 'n_cells_by_counts-0',
        'mean_counts-0', 'log1p_mean_counts-0', 'pct_dropout_by_counts-0',
        'total_counts-0', 'log1p_total_counts-0', 'gene_ids-0', 'feature_types-0', 'genome-0', 'feature_is_filtered-1', 
        'feature_name-1', 'feature_reference-1', 'feature_biotype-1', 'highly_variable', 'means', 'dispersions', 'dispersions_norm']:
        if v in adata.var.columns:
            del adata.var[v]
        
    for o in ['log1p_total_counts', 'mapped_reference_assembly', 'mapped_reference_annotation', 
        'alignment_software', 'donor_id', 'self_reported_ethnicity_ontology_term_id', 'donor_living_at_sample_collection', 
        'donor_menopausal_status', 'organism_ontology_term_id', 'sample_uuid', 'sample_preservation_method', 'tissue_ontology_term_id', 
        'development_stage_ontology_term_id', 'sample_derivation_process', 'sample_source', 'donor_BMI_at_collection', 
        'tissue_section_uuid', 'tissue_section_thickness', 'library_uuid', 'assay_ontology_term_id', 'sequencing_platform', 
        'is_primary_data', 'cell_type_ontology_term_id', 'disease_ontology_term_id', 'sex_ontology_term_id', 
        'nCount_Spatial', 'nFeature_Spatial', 'nCount_SCT', 'nFeature_SCT', 
        'suspension_type', 'tissue', 
        'self_reported_ethnicity', 'development_stage', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts',
        'pct_counts_in_top_50_genes', 'pct_counts_in_top_100_genes',
        'pct_counts_in_top_200_genes', 'pct_counts_in_top_500_genes',
        'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt']:
        if o in adata.obs.columns:
            del adata.obs[o]
            


with st.spinner('Loading Reference Scaffold...'):
    adata, pdata = build_scaffold(tissue)
    pdata.obsm['spatial'] = adata.obsm['spatial']
    pdata.raw = pdata
    pdata.X = pdata.X.astype(float)

with st.spinner('Integrating reference...'):
    adata_cyt = up.Run(name=tissue, 
        adatas=[adata], 
        adata_cm =adata, 
        lambda_s=1.0, 
        out='project', 
        ref_id=1, 
        outdir='./notebooks/output'
    )

    adata_cyt.obsm['latent'] = adata_cyt.obsm['project']
    adata = adata_cyt[adata_cyt.obs.source=='Breast CytAssist (Ref)']
    clean_adata(adata)
    
    
    st.caption('CytAssist Reference AnnData:')
    st.code(adata)


with st.spinner('Featurizing...'):
    gex = featurize(adata)
    pex = featurize(pdata, clr=False)
    d11 = adata.obsm['latent'].copy()
    d12 = pex.features
    


st.title(f'{tissue} (Human)')
st.info(f'🧬 Reference sample has {adata.obsm["latent"].shape[0]} spots, {adata.shape[1]} genes, and {pdata.shape[1]} proteins.')

with st.spinner(f'Loading pre-trained model for...'):
    model = InfoMaxVAE([d11.shape[1], pdata.shape[1]], latent_dim=64, encoder_dim=64)
    model.load_state_dict(torch.load(f'./notebooks/{tissue}_model.pth', map_location=torch.device('cpu')))

perf, examples, upload = st.tabs(["Model Performance", "Examples", "Upload"])

with upload:
    st.file_uploader('Upload your own ST data', type='h5ad')
            

with perf:
    with st.expander('View Model Architechture'):
        st.code(model)
        


import squidpy as sq

sample_shapes = [
    (1, (2103, 36503)),
    (2, (1400, 36503)),
    (3, (2364, 36503)),
    (4, (2504, 36503)),
    (5, (2694, 36503)),
    (6, (3037, 36503)),
    (7, (2086, 36503)),
    (8, (2801, 36503)),
    (9, (2694, 36503)),
    (10, (2473, 36503))
]


with examples:
    sel, run = st.columns(2)
    # sample_ids = sel.multiselect('Select tissue samples (10)', list(range(1, 11)))
    color_by = sel.selectbox('Color by', ['author_cell_type', 'cell_type'])
    
    cols = st.columns(5)
    sample_ids = [cols[0].checkbox(f'Sample {x}') for x in range(1, 6)] +\
        [cols[1].checkbox(f'Sample {x}') for x in range(6, 11)]
        
    sample_ids = [x+1 for x, y in enumerate(sample_ids) if y]

    if len(sample_ids) > 0:
        plots = st.columns(len(sample_ids))

    
    if st.button(f'🔗 Integrate & Run', disabled=len(sample_ids) == 0):
    
        with st.spinner('Integrating...'):

            start = time.perf_counter()
                
            aout = [project(
                        idx, 
                        adata_ref=adata, 
                        tissue=tissue) 
                    for idx in sample_ids]
                
        st.success(f'Integration completed in {((time.perf_counter()-start)/60.0):.2f} minutes.')

        
        for a in aout:
            clean_adata(a)
            st.code(a)

            with st.spinner('Imputing Surface Proteins...'):              
                gexa = featurize(a)
                A = gexa.adj_norm.to_dense()
                d11a = torch.tensor(a.obsm['latent']).float()
                imputed_proteins = model.impute(d11a, A)
                
                enc, _, _ = model.encoders[0].forward(d11a, A.nonzero().t().contiguous())
                z = model.fc_mus[0](enc, A).detach().numpy()
            
            left, right = st.columns(2)
            with st.spinner(f'Plotting UMAPs...'):   
                
                # embeddings = UMAP(n_components=2, n_neighbors=200, min_dist=0.1, random_state=42).fit_transform(z)
                embeddings = UMAP(n_components=2, n_neighbors=200, min_dist=0.1, random_state=42).fit_transform(imputed_proteins)
                
                
                plot_umap_grid(embeddings, imputed_proteins, pdata.var_names, None, size=50)
                st.pyplot(transparent=True, dpi=180)
                
                # plot_umap_grid(embeddings, a.to_df()[pdata.var_names].values, pdata.var_names, None, size=50)
                # st.pyplot(transparent=True, dpi=180)
                
            st.markdown('---')
            
            f, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=180)
            
            scatter = sns.scatterplot(
                embeddings[:, 0], 
                embeddings[:, 1], 
                hue=a.obs[color_by], 
                s=50, 
                edgecolor='black', 
                palette='Dark2',
                legend=True,
                ax = ax
            )
            
            ax.set_title('UMAP')
            ax.legend(ncols=2, loc='lower center', scatterpoints=1, fontsize=6, bbox_to_anchor=(0.5, -0.35))
            
            
            # fig = plt.gcf()
            # plt.axis('off')
            # plt.grid(b=None)
            right.pyplot(f, transparent=True, dpi=180)
            
            
            ix = a.obs.domain_id.unique()[0]
            
            adata_tmp = sc.read_h5ad(f'/ix/hosmanbeyoglu/kor11/CytAssist/Breast/visium/patient_{ix}.h5ad')
            
            f, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=180)
            sq.pl.spatial_scatter(adata_tmp, color=color_by, ax=ax, legend_loc=None, title='ST Data',
                    palette='Dark2', size=1.55, edgecolor='black', frameon=False, linewidth=0.35,figsize=(4, 4))
            left.pyplot(f, transparent=True, dpi=180)
        
            
            
        
        # with st.spinner('Plotting Integration Results...'):
        #     adata_all = AnnData.concatenate(*aout+[adata_cyt])
        #     red = cuml.UMAP(
        #         n_components=2,
        #         n_neighbors=100,
        #         min_dist=.1,
        #     )
        #     red.fit(adata_all[adata_all.obs.domain_id==0].obsm['latent'])

        #     plots = st.columns(len(sample_ids))
            # targets = adata_all[adata_all.obs.domain_id==0]
            # plot_data = red.transform(targets.obsm['project'])
            # scatter = sns.scatterplot(
            #     plot_data[:, 0], 
            #     plot_data[:, 1], 
            #     color='red', 
            #     s=30, 
            #     legend=False, 
            #     edgecolor='white', 
            # )
                
            # fig = plt.gcf()
            # plt.axis('off')
            # plt.grid(b=None)
            # plots[0].pyplot(fig, transparent=True)
                
            # for ix, plot in zip(sample_ids, plots):
            #     targets = adata_all[adata_all.obs.domain_id==ix]
            #     plot_data = red.transform(targets.obsm['project'])
            #     scatter = sns.scatterplot(
            #         plot_data[:, 0], 
            #         plot_data[:, 1], 
            #         hue=targets.obs.author_cell_type, 
            #         s=30, 
            #         legend=False, 
            #         edgecolor='black', 
            #         palette='Dark2'
            #     )
                

                
            #     fig = plt.gcf()
            #     plt.axis('off')
            #     plt.grid(b=None)
            #     plot.pyplot(fig, transparent=True)
            
