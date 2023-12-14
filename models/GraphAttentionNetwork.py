import torch 
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer

class GraphAttentionNetwork(nn.Module):
  def __init__(self, input_dim, output_dim, num_hidden_layers, num_heads, alpha, dropout):
    super().__init__()
    self.attention_layers = [
      GraphAttentionLayer(input_dim, output_dim, alpha, dropout) 
      for _ in range(num_hidden_layers)
    ]
    for i, attention in enumerate(self.attention_layers):
      self.add_module(f'attention_{i}', attention)

    self.out_attention = GraphAttentionLayer(num_hidden_layers*num_heads, output_dim, alpha, dropout)
    self.droupout = dropout

  def forward(self, x, adj):
    x = F.dropout(x, self.dropout)
    x = torch.cat([attention_layer(x, adj) for attention_layer in self.attention_layers], dim=-1)
    x = F.dropout(x, self.dropout)
    x = F.leaky_relu(self.out_ttention(x, adj))

    return F.log_softmax(x, dim=1)
    
