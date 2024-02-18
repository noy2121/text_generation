import torch
import torch.nn as nn
import torch.nn.functional as F

from.multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, nheads, ff_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(model_dim, nheads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # add & norm
        x = self.fc2(self.relu(self.fc1(x)))  # feed-forward
        x = self.norm2(x + self.dropout(x))  # add & norm

        return x

