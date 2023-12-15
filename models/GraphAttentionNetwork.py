import torch 
import torch.nn as nn
import torch.nn.functional as F
from layers.GraphAttentionLayer import GraphAttentionLayer

class GraphAttentionNetwork(nn.Module):
  def __init__(self, input_dim, output_dim, num_hidden_layers, num_heads, alpha, dropout):
    super().__init__()
    # k-head attention first layer
    self.attention_layers = [
      GraphAttentionLayer(input_dim, num_hidden_layers, alpha, dropout) 
      for _ in range(num_heads)
    ]
    for i, attention in enumerate(self.attention_layers):
      self.add_module(f'attention_{i}', attention)
    # second layer
    self.out_attention = GraphAttentionLayer(num_hidden_layers*num_heads, output_dim, alpha, dropout)
    self.dropout = dropout

  def forward(self, x, adj):
    x = F.dropout(x, self.dropout)
    x = torch.cat([attention_layer(x, adj) for attention_layer in self.attention_layers], dim=-1)
    x = F.dropout(x, self.dropout)
    x = F.elu(self.out_attention(x, adj))
    return x
    
