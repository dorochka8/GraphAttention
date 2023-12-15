# Graph Attention Netoworks

Here is presented the implementation of paper by Petar Veličković et al., `Graph Attention Networks` (ICLR 2018): \
https://arxiv.org/pdf/1710.10903.pdf


## Overview
Before start working, please make sure that you have installed all required libraries (see `requirements.txt`).

Setups are in `config.py`.

### Cora dataset 
Citation dataset, 2708 nodes (papers). 7 classes:
  - (0) Theory;
  - (1) Reinforcement_Learning;
  - (2) Genetic_Algorithms;
  - (3) Neural_Networks;
  - (4) Probabilistic_Methods;
  - (5) Case_Based;
  - (6) Rule_Learning.

## Training
Training process takes less than a minute (to pass 1000 epochs) and last train accuracy is 0.879, last validation accuracy is 0.442. \
<img src="https://github.com/dorochka8/GraphAttention/assets/97133490/39c85abd-fc05-4e77-a3d0-4f9e5ab9f525" width="600" height="500">
