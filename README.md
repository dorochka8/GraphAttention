# Graph Attention Netoworks

This repository is dedicated to the implementation of the "Graph Attention Networks" paper by Petar Veličković et al., (ICLR 2018): \
https://arxiv.org/pdf/1710.10903.pdf \
Graph Attention Networks introduce an attention-based architecture for node classification in graph-structured data, offering advancements in various applications like social networks analysis and bioinformatics.

## Installation and Configurations
To get started with this project, please ensure you have all the necessary libraries installed. Refer to the requirements.txt file for a complete list. You can install these packages using the command:
```
pip install -r requirements.txt
```
Customize your training and model parameters in config.py. This file includes settings for learning rate, number of epochs, and other model-specific configurations that you can tweak according to your needs. 

All datasets are citation networks from the paper https://arxiv.org/pdf/1603.08861.pdf. 
- **Cora dataset**: consists of 2708 nodes (papers), each with 1433 features classesied into 7 classes including "Reinforcement_Learning", "Neural_Networks", etc.
- **CiteSeer dataset**: consists of 3327 nodes (papers), each with 3703 features classified into 6 classes including "Artificial_Intelligence", "Machine_Learning", etc.
- **PubMed dataset**: consists of 19717 nodes (papers), each with 500 features classified into 3 classes related to Diabetes Mellitus.

## Getting Started 
To train the model, navigate to 'training' directory and run 'main.py':
```
cd training
python main.py
```

## Results
Training process takes less than a minute (to pass 1000 epochs).After training, you can view accuracy of the model on each dataset. For instance: \
Cora last train accuracy is 0.879, last validation accuracy is 0.442. \
<img src="https://github.com/dorochka8/GraphAttention/assets/97133490/34b21715-c3c9-47e0-8ef9-6f816e967f4f" width="600" height="500"> \
CiteSeer last train accuracy is 0.950, last validation accuracy is 0.400. \
<img src="https://github.com/dorochka8/GraphAttention/assets/97133490/6d0aa06c-b0c0-4ddd-a58e-e39e64ae7422" width="600" height="500"> \
PubMed last train accuracy is 1.000, last validation accuracy is 0.606. \
<img src="https://github.com/dorochka8/GraphAttention/assets/97133490/970d79be-7085-4561-a55d-c33462ea0bc3" width="600" height="500">
