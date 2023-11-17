import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphAttentionLayer import GraphAttentionLayer

class MultiHeadGraphAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, alpha=0.2, dropout=0.0, concat=True):
        super().__init__()
        self.heads = nn.ModuleList()
        self.concat  = concat
        self.dropout = dropout
    
        for i in range(num_heads):
            self.heads.append(
                GraphAttentionLayer(input_dim, output_dim, alpha=alpha, dropout=self.dropout)
            )


    def forward(self, x, adj):
        if self.concat:
            return torch.cat([head(x, adj) for head in self.heads], dim=1)
        
        else:
            return torch.mean(torch.stack([head(x, adj) for head in self.heads]), dim=0)
