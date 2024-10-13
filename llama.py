import math

from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from pos_embed import *
from module_utils import *



def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:

    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class LLAMA(nn.Module):
    def __init__(self,in_dim,in_frame,hidden_dim,n_heads,num_layer=2):
        super(LLAMA,self).__init__()
        self.freqs_cis=precompute_freqs_cis(hidden_dim*n_heads,in_dim)
        self.time_norm=RMSNorm(hidden_dim*n_heads)
        self.in_proj=nn.Linear(in_frame,hidden_dim*n_heads)
        self.model = []
        for i in range(0,num_layer):
            self.model.append(torch.nn.Sequential(Res_attention(hidden_dim*n_heads,1,hidden_dim*n_heads,self.freqs_cis)))
        self.model=torch.nn.ModuleList(self.model)
    def forward(self,x):
        x=self.in_proj(x)
        for layer in self.model:
            x=layer(x)
        return x
 