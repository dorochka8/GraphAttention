import os, sys
config_path = '/your/path/to/the/GraphAttention/'
sys.path.append(config_path)

import glob
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 

from config import config
from preprocess_data import accuracy, modify_adj


def training(model, X, y, idx_train, idx_val, adj, optimizer, epochs, dataset_name):
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
        if loss_trains[-1] < best: 
            best = loss_trains[-1]
            best_epoch = epoch+1
            counter = 0
        else:
            counter += 1
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
    fig.suptitle(f'{dataset_name} dataset')
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
