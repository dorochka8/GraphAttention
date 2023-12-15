import numpy as np 
import scipy as sp
import torch
from torch_geometric.datasets import Planetoid


def do_normalized_features(features):
    row_sum = np.array(np.sum(features, axis=1))
    row_inv_sqrt = (1 / row_sum).flatten()
    row_inv_sqrt[np.isinf(row_inv_sqrt)] = 0.0
    row_inv_sqrt = sp.diags(row_inv_sqrt)
    features = row_inv_sqrt @ features
    return features
    

def do_normalized_adjacency(edges, labels):
    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0, :], edges[1, :])), 
                        shape=(labels.shape[0], labels.shape[0]), 
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + np.eye(adj.shape[0])

    # normalize adjacency matrix by rows
    row_sum = np.array(np.sum(adj, axis=1))
    row_inv_sqrt = 1 / (row_sum**0.5).flatten()
    row_inv_sqrt[np.isinf(row_inv_sqrt)] = 0.0
    row_inv_sqrt = sp.diags(row_inv_sqrt)
    adj_matrix_normalized = adj @ row_inv_sqrt.T @ row_inv_sqrt
    return adj_matrix_normalized
    

def extracted_info(path, dataset):
    print(f'Loading {dataset} dataset...')    
    data = Planetoid(root=path, name=dataset)[0]
    features = torch.FloatTensor(
        np.array(
            do_normalized_features(
                sp.csr_matrix(data['x'], dtype=np.float32)
            ).todense()
        )
    )
    print(f'\tFeatures normalized ✓')
    
    labels = data['y']
    edges = data['edge_index']
    adj = torch.FloatTensor(np.array(do_normalized_adjacency(edges, labels)))
    print(f'\tNormalized adjacency matrix created ✓')
    
    idx_train = data['train_mask']
    idx_val   = data['val_mask']
    idx_test  = data['test_mask']
    return idx_train, idx_val, idx_test, adj, features, labels


def idx_finder(range_to_cut):
    start, finish = None, None
    for i, idx in enumerate(range_to_cut):
        if idx.item() == True and start == None:
            start = i
        if idx.item() == False and finish == None and start != None:
            finish = i
            # stop not to waste time 
            break
    return start, finish


def modify_adj(adj, idx_to_keep):
    start, finish = idx_finder(idx_to_keep)
    modified_adj = adj[start:finish, start:finish]
    return modified_adj
    

def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=-1)
    correct = (preds == labels).sum()
    return correct / len(labels)
