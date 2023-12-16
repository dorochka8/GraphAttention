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
To start training the model open `training -> main.py`.

Training process takes less than a minute (to pass 1000 epochs). \
Cora last train accuracy is 0.879, last validation accuracy is 0.442. \
<img src="https://github.com/dorochka8/GraphAttention/assets/97133490/34b21715-c3c9-47e0-8ef9-6f816e967f4f" width="600" height="500"> \
CiteSeer last train accuracy is 0.950, last validation accuracy is 0.400. \
<img src="https://github.com/dorochka8/GraphAttention/assets/97133490/6d0aa06c-b0c0-4ddd-a58e-e39e64ae7422" width="600" height="500"> \
PubMed last train accuracy is 1.000, last validation accuracy is 0.606. \
<img src="https://github.com/dorochka8/GraphAttention/assets/97133490/970d79be-7085-4561-a55d-c33462ea0bc3" width="600" height="500">\
