# Graph Attention Netoworks

Here is presented the implementation of paper by Petar Veličković et al., `Graph Attention Networks` (ICLR 2018): \
https://arxiv.org/pdf/1710.10903.pdf


## Overview
Before start working, please make sure that you have installed all required libraries (see `requirements.txt`).

Setups are in `config.py`.


## Training
- Cora dataset
  Citation dataset, 2708 nodes (papers). \
  7 classes:
    - (0) Theory;
    - (1) Reinforcement_Learning;
    - (2) Genetic_Algorithms;
    - (3) Neural_Networks;
    - (4) Probabilistic_Methods;
    - (5) Case_Based;
    - (6) Rule_Learning.
Most papers belongs to 3rd class.

Training process takes less than a minute (to pass 1000 epochs) and train accuracy is 0.87857, while validation accuracy is 0.442. 
