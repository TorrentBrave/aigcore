"""
Visualizing the Self-Attention Mechanism - https://codingowen.github.io/blog/2025/02/27/self_attention_intuition/
Building the Self-Attention Mechanism from scratch
    (1) https://codingowen.github.io/projects/self_attention/
    (2) https://mohdfaraaz.medium.com/implementing-self-attention-from-scratch-in-pytorch-776ef7b8f13e
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d, d_k, d_q, d_v):
        super(SelfAttention, self).__init__()
        self.d = d
        self.d_k = d_k
        self.d_q = d_q
        self.d_v = d_v

        self.W_K = nn.Parameter(torch.rand(d, d_k))
        self.W_Q = nn.Parameter(torch.rand(d, d_q))
        self.W_V = nn.Parameter(torch.rand(d, d_v))

    def forward(self, X):
        K = X @ self.W_K
        Q = X @ self.W_Q
        V = X @ self.W_V

        attention_scores = Q @ K.T / math.sqrt(self.d_k)
        attention_weights = F.softmax(attention_scores, dim=-1)
        context_vector = attention_weights @ V

        return context_vector
