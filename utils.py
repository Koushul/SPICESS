from matplotlib import pyplot as plt
import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors
import torch
import scipy.sparse as sp
from anndata import AnnData
import networkx as nx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
        source: https://github.com/zfjsail/gae-pytorch/blob/master/gae/utils.py"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

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

def plotA(coord,adj):
    g=nx.from_numpy_matrix(adj)
    pos={}
    for n in range(coord.shape[0]):
        pos[n]=(coord[n][0],coord[n][1])
    fig, ax = plt.subplots(dpi=200)
    nx.draw(g, pos, node_size=6, width=0.5)
    plt.show()

    
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