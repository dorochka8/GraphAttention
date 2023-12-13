import numpy as np 

import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess_data import extracted_info, accuracy

def training(model, X, y, idx_train, idx_val, adj, optimizer, epochs):
    t_total = time.time()
    loss_values = []
    best, best_epoch = 0.0, 0
    model.train()
    for epoch in range(epochs):
        epoch_time = time.time()
        optimizer.zero_grad()
        outputs = model(X[idx_train], adj)
        loss_train = F.nll_loss(outputs[idx_train], y[idx_train])
        acc_train  = accuracy(outputs, y[idx_train])
        loss_values.append(loss_train)

        loss_train.backward()
        optimizer.step()
        
        with torch.no_grad():
            model.eval()
            outputs  = model(X, adj)
            loss_val = F.nll_loss(outputs[idx_val], y[idx_val])
            acc_val  = accuracy(outputs[idx_val], y[idx_val])
    
        print(f'Epoch: {epoch+1}')
        print(f'Train_loss: {loss_train.data.item():.5f} \tTrain_acc: {acc_train.data.item():.5f}')
        print(f'Val_loss:   {loss_val.data.item():.5f} \tVal_acc: {acc_val.ata.item():.5f}')
        print(f'time: {(time.time()-epoch_time):.5f}')

        torch.save(model.state_dict(), f'{epoch+1}.pkl')
        if loss_values[-1] < best: 
            best = loss_values[-1]
            best_epoch = epoch+1
            counter = 0
        else:
            counter += 1
        if counter == 100:
            break

device = 'cuda' if torch.cuda.is_available else 'cpu'

random.seed(46911356)
np.random.seed(46911356)
torch.manual_seed(46911356)
if device == 'cuda':
    torch.cuda.manual_seed(46911356)

