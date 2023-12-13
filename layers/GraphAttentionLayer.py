import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    GraphAttentionLayer (similar to, p.3 https://arxiv.org/pdf/1710.10903.pdf)

    input:  a set of node features h with `input_dim` equals the number of features per node
    return: a set of -new- node features of different cardinality of `output_dim`
    """
    def __init__(self, input_dim, output_dim, alpha=0.2, dropout=0.0):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.alpha   = alpha    
        self.dropout = dropout
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
        self.attention_mechanism = nn.Parameter(torch.randn(2 * output_dim, 1))


    def forward(self, x, adj):
        x = torch.mm(x, self.weights) # x.shape: [N, output_dim], N - number of nodes
        e = self._do_attention_mech_input(x)
        
        # zero_vec to make the further usage softmax interpritable 
        # (if left with zeros, they would contribute to the softmax calculation
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        h_prime = torch.matmul(attention, x)

        return F.leaky_relu(h_prime, negative_slope=self.alpha)
    

    def _do_attention_mech_input(self, x):
        w1 = torch.matmul(x, self.attention_mechanism[:self.output_dim, :])
        w2 = torch.matmul(x, self.attention_mechanism[self.output_dim:, :])
        e = w1 + w2.T
        return F.leaky_relu(e) 
