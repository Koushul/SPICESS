from argparse import Namespace
from matplotlib import pyplot as plt
import numpy as np
import scipy
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
import torch
import scipy.sparse as sp
from anndata import AnnData
import networkx as nx
from scipy.sparse import csr_matrix
import math
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, f1_score
import gudhi
import scanpy as sc

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def getA_knn(sobj_coord_np, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(sobj_coord_np)
    a=nbrs.kneighbors_graph(sobj_coord_np, mode='connectivity')
    a=a-sp.identity(sobj_coord_np.shape[0], format='csr')
    return a

def clr_normalize_each_cell(adata: AnnData, inplace: bool = False):
    """Take the logarithm of the surface protein count for each cell, 
    then center the data by subtracting the geometric mean of the counts across all cells."""
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else adata.X)
    )
    
    return adata

    
def mask_nodes_edges(nNodes,testNodeSize=0.1,valNodeSize=0.05,seed=3):
    # randomly select nodes; mask all corresponding rows and columns in loss functions
    np.random.seed(seed)
    num_test=int(round(testNodeSize*nNodes))
    num_val=int(round(valNodeSize*nNodes))
    all_nodes_idx = np.arange(nNodes)
    np.random.shuffle(all_nodes_idx)
    test_nodes_idx = all_nodes_idx[:num_test]
    val_nodes_idx = all_nodes_idx[num_test:(num_val + num_test)]
    train_nodes_idx=all_nodes_idx[(num_val + num_test):]
    
    return torch.tensor(train_nodes_idx),torch.tensor(val_nodes_idx),torch.tensor(test_nodes_idx)

def train_test_split(adata):
    adata.obs['idx'] = list(range(0, len(adata)))
    xy = adata.obsm['spatial']
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
    

def graph_alpha(spatial_locs, n_neighbors=10):
    """
    Construct a geometry-aware spatial proximity graph of the spatial spots of cells by using alpha complex.
    """
    A_knn = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode='distance')
    estimated_graph_cut = A_knn.sum() / float(A_knn.count_nonzero())
    spatial_locs_list = spatial_locs.tolist()
    n_node = len(spatial_locs_list)
    alpha_complex = gudhi.AlphaComplex(points=spatial_locs_list)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=estimated_graph_cut ** 2)
    skeleton = simplex_tree.get_skeleton(1)
    initial_graph = nx.Graph()
    initial_graph.add_nodes_from([i for i in range(n_node)])
    for s in skeleton:
        if len(s[0]) == 2:
            initial_graph.add_edge(s[0][0], s[0][1])

    extended_graph = nx.Graph()
    extended_graph.add_nodes_from(initial_graph)
    extended_graph.add_edges_from(initial_graph.edges)

    # Remove self edges
    for i in range(n_node):
        try:
            extended_graph.remove_edge(i, i)
        except:
            pass

    return nx.to_scipy_sparse_matrix(extended_graph, format='csr')

def featurize(input_adata, neighbors=6, clr=False, normalize_total=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    varz = Namespace()
    features = input_adata.copy()
    if normalize_total:
        sc.pp.normalize_total(features, target_sum=1e4)
    train_test_split(features)
    features.X = csr_matrix(features.X)
    features.X = features.X.toarray()
    features_raw = torch.tensor(features.X)

    if clr:
        features = clr_normalize_each_cell(features)
        
    featurelog = np.log2(features.X+1/2)
    scaler = MinMaxScaler()
    featurelog = np.transpose(scaler.fit_transform(np.transpose(featurelog)))
    feature = torch.tensor(featurelog)

    # adj = getA_knn(features.obsm['spatial'], neighbors)
    adj = graph_alpha(features.obsm['spatial'], n_neighbors=neighbors)
    varz.adj_empty = adj
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = torch.tensor(adj_label.toarray()).to(device)
    pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()).to(device)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    adj_norm = preprocess_graph(adj).to(device)

    feature = feature.float().to(device)
    adj_norm = adj_norm.float().to(device)
    maskedgeres= mask_nodes_edges(features.shape[0], testNodeSize=0, valNodeSize=0.1)
    varz.train_nodes_idx, varz.val_nodes_idx, varz.test_nodes_idx = maskedgeres
    features_raw = features_raw.to(device)

    coords = torch.tensor(features.obsm['spatial']).float()
    sp_dists = torch.cdist(coords, coords, p=2)
    
    
    varz.edge_index = sparse_mx_to_torch_edge_list(adj).to(device)
    varz.sp_dists = torch.div(sp_dists, torch.max(sp_dists)).to(device)
    varz.features = feature
    varz.features_raw = features_raw
    varz.adj_norm = adj_norm
    varz.norm = norm
    varz.pos_weight = pos_weight
    varz.adj_label = adj_label
    varz.device = device
    
    return varz

def update_vars(v1, v2):
    v1 = vars(v1)
    v1.update(vars(v2))
    return Namespace(**v1)


def feature_corr(preds, targets):
    assert preds.shape == targets.shape
    return [spearmanr(targets[:, i], preds[:, i]).statistic \
                for i in range(targets.shape[1])]
    


def adj_auc(adj, adj_predicted) -> float:
    return roc_auc_score(adj.flatten(), adj_predicted.flatten())
    
    
def adj_f1(adj, adj_predicted) -> float:
    return f1_score(adj.flatten(), adj_predicted.flatten())


from sklearn.metrics import precision_score, recall_score

def calculate_precision_recall(adj, adj_predicted):
    precision = precision_score(adj.flatten(), adj_predicted.flatten())
    recall = recall_score(adj.flatten(), adj_predicted.flatten())
    return precision, recall



def select_points_for_rectangle(points, bottom_left, top_right):
    """
    Select points that are within the area of a rectangle.

    Parameters:
    points (list of tuples): The points to select from.
    bottom_left (tuple): The coordinates of the bottom left corner of the rectangle.
    top_right (tuple): The coordinates of the top right corner of the rectangle.

    Returns:
    list of tuples: The selected points.
    """
    selected_points = []
    for point in points:
        if bottom_left[0] <= point[0] <= top_right[0] and bottom_left[1] <= point[1] <= top_right[1]:
            selected_points.append(point)
    return selected_points


def select_points_for_concentric_circles(points, center, inner_radius, outer_radius):
    """
    Select points that are within the area of two concentric circles.

    Parameters:
    points (list of tuples): The points to select from.
    center (tuple): The center of the circles.
    inner_radius (float): The radius of the inner circle.
    outer_radius (float): The radius of the outer circle.

    Returns:
    list of tuples: The selected points.
    """
    selected_points = []
    for point in points:
        distance_from_center = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
        if inner_radius <= distance_from_center <= outer_radius:
            selected_points.append(point)
    return selected_points


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