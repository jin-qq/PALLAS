import math

from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def precompute_freqs_cis(dim, end, base=10000.0):
    theta = 1.0 / (base ** (torch.arange(0, dim, 2) / dim)) #dim//2
    m = torch.arange(end)                                   #end
    freqs = torch.outer(m, theta)                          #end,dim//2
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
    return freqs_cis 


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
#rotate PE, forks from https://github.com/meta-llama/llama3/blob/main/llama/model.py


class TrainablePositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, d_model):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        pe = nn.Embedding(self.max_sequence_length, self.d_model)
        nn.init.constant(pe.weight, 0.)
        return pe
    
#trainable PE from vision transformer

def Position_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    position = np.arange(0, max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return torch.tensor(pe, dtype=torch.float32)