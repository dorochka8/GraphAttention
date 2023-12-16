# Graph Attention Netoworks

Here is presented the implementation of paper by Petar Veličković et al., `Graph Attention Networks` (ICLR 2018): \
https://arxiv.org/pdf/1710.10903.pdf


## Overview
Before start working, please make sure that you have installed all required libraries (see `requirements.txt`).

Setups are in `config.py`. \
All datasets are citation networks from the paper https://arxiv.org/pdf/1603.08861.pdf. 
### Cora dataset:
2708 nodes (papers), 1433 features, 7 classes:
  - (0) Theory;
  - (1) Reinforcement_Learning;
  - (2) Genetic_Algorithms;
  - (3) Neural_Networks;
  - (4) Probabilistic_Methods;
  - (5) Case_Based;
  - (6) Rule_Learning.
### CiteSeer dataset:
3327 nodes (papers), 3703 features, 6 classes:
  - (0) Agents;
  - (1) Artificial_Intelligence;
  - (2) Database;
  - (3) Human_Computer_Interaction;
  - (4) Machine_Learning;
  - (5) Information_Retrieval.
### PubMed dataset: 
19717 nodes (papers), 500 features, 3 classes:
  - (0) Diabetes_Mellitus_Experimental;
  - (1) Diabetes_Mellitus_Type1;
  - (2) Diabetes_Mellitus_Type2.

## Training
To start training the model do `training -> main.py`.

Training process takes less than a minute (to pass 1000 epochs). \
Cora last train accuracy is 0.879, last validation accuracy is 0.442. \
<img src="https://github.com/dorochka8/GraphAttention/assets/97133490/34b21715-c3c9-47e0-8ef9-6f816e967f4f" width="600" height="500"> \
CiteSeer last train accuracy is 0.950, last validation accuracy is 0.400. \
<img src="https://github.com/dorochka8/GraphAttention/assets/97133490/6d0aa06c-b0c0-4ddd-a58e-e39e64ae7422" width="600" height="500"> \
PubMed last train accuracy is 1.000, last validation accuracy is 0.606. \
<img src="https://github.com/dorochka8/GraphAttention/assets/97133490/970d79be-7085-4561-a55d-c33462ea0bc3" width="600" height="500">\
