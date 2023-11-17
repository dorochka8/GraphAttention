import torch.nn as nn

config = {
    # optimizer
    'lr': 5e-3,
    'weight_decay': 5e-4,

    # loss function 
    'loss_fn': nn.CrossEntropyLoss(),

    # model set-up
    'model_alpha'      : 0.2,   # 
    'model_dropout'    : 0.6,   #  
    'model_concat'     : True,  # concatenation of attention outputs if False -> *averaging*
    'model_num_heads'  : [8, 1],# number of attention heads
    'model_num_hidden' : 8,     # number of hidden layers
    'train_epochs'     : 200,   # total number of epochs to train
    'seed'             : 72     
    }
