config = {
    # optimizer parameters
    'lr': 5e-3,
    'weight_decay': 5e-4,

    # model architecture  
    'model_alpha'      : 0.2, # slope ratio for ELU   
    'model_dropout'    : 0.6,     
    'model_num_heads'  : 8,   # number of attention heads
    'model_num_hidden' : 8,   # number of hidden layers
    
    # model training
    'train_epochs'     : 1000, # total number of epochs to train
    'seed'             : 46911356,
    'patience'         : 1000, 
    }
