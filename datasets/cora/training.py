# import os, sys
# config_path = '/your/path/to/the/GraphAttention/'
# sys.path.append(config_path)

import glob
import numpy as np 
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from models.GraphAttentionNetwork import GraphAttentionNetwork
from preprocess_data import extracted_info, accuracy, modify_adj


def training(model, X, y, idx_train, idx_val, adj, optimizer, epochs):
    t_total = time.time()
    loss_trains, acc_trains = [], []
    loss_vals, acc_vals = [], []
    best, best_epoch, counter = 0.0, 0, 0
    model.train()
    for epoch in range(epochs):
        epoch_time = time.time()
        optimizer.zero_grad()
        
        outputs = model(X[idx_train], modify_adj(adj, idx_train))
        loss_train = F.cross_entropy(outputs, y[idx_train])
        acc_train  = accuracy(outputs, y[idx_train])
        loss_trains.append(loss_train.item())
        acc_trains.append(acc_train.item())

        loss_train.backward()
        optimizer.step()
        
        with torch.no_grad():
            model.eval()
            outputs  = model(X[idx_val], modify_adj(adj, idx_val))
            loss_val = F.cross_entropy(outputs, y[idx_val])
            acc_val  = accuracy(outputs, y[idx_val])
            loss_vals.append(loss_val.item())
            acc_vals.append(acc_val.item())
    
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}')
            print(f'Train_loss: {loss_train.data.item():.5f} \tTrain_acc: {acc_train.data.item():.5f}')
            print(f'Val_loss:   {loss_val.data.item():.5f} \tVal_acc: {acc_val.data.item():.5f}')
            print(f'time: {(time.time()-epoch_time):.5f}')

        torch.save(model.state_dict(), f'{epoch+1}.pkl')
        if loss_values[-1] < best: 
            best = loss_values[-1]
            best_epoch = epoch+1
            counter = 0
        else: counter += 1
        if counter == config['patience']:
            break
            
        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
            
    print(f'time spent: {(time.time()-t_total):.5f}')
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('"Cora" dataset')
    axs[0, 0].plot(loss_trains)
    axs[0, 0].set_title('Train loss', fontsize=10)
    axs[0, 1].plot(acc_trains, 'tab:orange')
    axs[0, 1].set_title('Train accuracy', fontsize=10)
    axs[1, 0].plot(loss_vals)
    axs[1, 0].set_title('Validation loss', fontsize=10)
    axs[1, 1].plot(acc_vals, 'tab:orange')
    axs[1, 1].set_title('Validation accuracy', fontsize=10)
    for ax in axs.flat:
        ax.set_xlabel('epochs', fontsize=8)
        ax.tick_params(labelsize=8)
    plt.show()
    return 


device = 'cuda' if torch.cuda.is_available else 'cpu'
random.seed(config['seed'])
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
if device == 'cuda':
    torch.cuda.manual_seed(config['seed'])
    
# download the dataset 
path = "./data"  # change manually to the dir where the data downloaded
os.makedirs(path, exist_ok=True)
dataset_name = 'Cora'
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

training(model, features, labels, idx_train, idx_val, adj, optimizer, config['train_epochs'])
