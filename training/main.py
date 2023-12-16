import os, sys
config_path = '/your/path/to/the/GraphAttention/'
sys.path.append(config_path)

import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid

from training import training
from models.GraphAttentionNetwork import GraphAttentionNetwork
from preprocess_data import extracted_info
from config import config 


device = 'cpu' #'cuda' if torch.cuda.is_available else 'cpu'

random.seed(config['seed'])
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
if device == 'cuda':
    torch.cuda.manual_seed(config['seed'])


path = "./data"
os.makedirs(path, exist_ok=True)
dataset_names = ['Cora', 'CiteSeer', 'PubMed']
# Choose dataset from the above to test 
dataset_name = dataset_names[1]
dataset = Planetoid(root=path, name=dataset_name)
data = dataset[0]

idx_train, idx_val, idx_test, adj, features, labels = extracted_info(path, dataset_name)
model = GraphAttentionNetwork(
    input_dim=features.shape[1], 
    output_dim=len(torch.unique(labels)),
    num_hidden_layers=config['model_num_hidden'], 
    num_heads=config['model_num_heads'], 
    alpha=config['model_alpha'],
    dropout=config['model_dropout'], 
                            )

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'] )
if device == 'cuda':
    model.cuda()
    idx_train, idx_val, idx_test = idx_train.cuda(), idx_val.cuda(), idx_test.cuda()
    adj, features, labels = adj.cuda(), features.cuda(), labels.cuda()


training(model, features, labels, idx_train, idx_val, adj, optimizer, config['train_epochs'], dataset_name)
