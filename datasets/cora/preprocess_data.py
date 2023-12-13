import numpy as np 
import scipy as sp
import torch

def do_onehot(labels):
    unique_classes = sorted(list(set(labels)))
    classes_dict = {
        classs: np.identity(len(unique_classes))[i, :] 
        for i, classs in enumerate(unique_classes)
    }
    onehot_labels = np.array(list(map(
        classes_dict.get, labels
    )), dtype=np.int32)
    return onehot_labels

def do_normalized_adjacency(edges, labels):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), 
                        shape=(labels.shape[0], labels.shape[0]), 
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + np.eye(adj.shape[0])

    # normalize adjacency matrix by rows
    row_sum = sum(adj, dim=1)
    row_inv_sqrt = 1 / (row_sum**0.5).flatten()
    row_inv_sqrt[float('inf')] = 0.0
    row_inv_sqrt = sp.diags(row_inv_sqrt)
    adj_matrix_normalized = adj @ row_inv_sqrt.T @ row_inv_sqrt
    return adj_matrix_normalized

def do_normalized_features(features):
    row_sum = sum(features, dim=1)
    row_inv_sqrt = 1 / (row_sum).flatten()
    row_inv_sqrt[float('inf')] = 0.0
    row_inv_sqrt = sp.diags(row_inv_sqrt)
    features = row_inv_sqrt @ features
    return features

def extracted_info(path, dataset):
    print(f'Loading {dataset} dataset')
    feat_labels = np.genfromtxt(f'{path}{dataset}.content', dtype=np.dtype(str))
    features = torch.FloatTensor(
        np.array(
            do_normalized_features(
                sp.csr_matrix(feat_labels[:, 1:-1], dtype=np.float32)
            ).todense()
        )
    )
    labels = torch.LongTensor(
        np.where(
            do_onehot(feat_labels[:, -1])[1]
        )
    )

    idx = np.array(feat_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.getfromtxt(f'{path}{dataset}', dtype=np.int32)
    edges = np.array(list(map(
        idx_map.get, edges_unordered.flatten()
        )), dtype=np.int32).reshape(edges_unordered.shape)
    adj = torch.FloatTensor(
        np.array(
            do_normalized_adjacency(edges, labels).todense()
        )
    )

    idx_train = torch.LongTensor(range(140))
    idx_val   = torch.LongTensor(range(200, 500))
    idx_test  = torch.LongTensor(range(500, 1500))

    return idx_train, idx_val, idx_test, adj, features, labels

def accuracy(outputs, labels):
    preds = outputs.max(1)[1].type_as(labels)
    correct = ((preds == labels).double()).sum()
    return correct / len(labels)
    
