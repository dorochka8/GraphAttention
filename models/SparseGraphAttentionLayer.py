import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseGraphAttention(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha  = alpha
        self.dropout = dropout
    
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
        self.attention_mechanism = nn.Parameter(torch.randn(2 * output_dim, 1))


    def forward(self, x, adj):
        edges = adj.nonzero().T() 
        # catch the number of samples
        N = x.size()[0]               
        
        hidden_output = x @ self.weights
        edge_hidden_output = torch.cat(
            (hidden_output[edges[0, :], :],
             hidden_output[edges[1, :], :]),
             dim=1
        ).T()
        edge_exponential_hidden_output = torch.exp(
            (-self.leakyrelu(
                self.attention_mechanism @ edge_hidden_output
                             ).squeeze()
            )
        )
