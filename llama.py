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
    def __init__(self,in_dim,in_frame,hidden_dim,n_heads,out_dim,num_ik=4,num_layer=2):
        super(LLAMA,self).__init__()
        self.freqs_cis=precompute_freqs_cis(hidden_dim*n_heads,in_dim)
        self.spa_freqs_cis=precompute_freqs_cis(hidden_dim*n_heads,hidden_dim*n_heads)
        self.time_norm=RMSNorm(hidden_dim*n_heads)
        self.time_in_proj=nn.Linear(in_frame,hidden_dim*n_heads)
        self.spa_in_proj=nn.Linear(in_dim,hidden_dim*n_heads)
        self.time_model = []
        self.spa_model=[]
        for i in range(0,num_layer):
            self.time_model.append(torch.nn.Sequential(Res_attention(hidden_dim*n_heads,1,hidden_dim*n_heads,self.freqs_cis)))
            self.spa_model.append(torch.nn.Sequential(Res_attention(hidden_dim*n_heads,1,hidden_dim*n_heads,self.spa_freqs_cis)))
        self.time_model=torch.nn.ModuleList(self.time_model)
        self.spa_model=torch.nn.ModuleList(self.spa_model)
        
        self.spa_out_proj=nn.Linear(hidden_dim*n_heads,out_dim)
        self.time_out_proj=nn.Linear(hidden_dim*n_heads,in_frame)
        
        self.ik=RealNVP(num_ik,out_dim,out_dim)
    def forward(self,x):
        x=self.time_in_proj(x)
        for layer in self.time_model:
            x=layer(x)
        x=self.spa_in_proj(x.permute(0,2,1))
        for layer in self.spa_model:
            x=layer(x)
        x=self.spa_out_proj(x)
        x=self.time_out_proj(x.permute(0,2,1))
        rot=self.ik(x.permute(0,2,1))[0].permute(0,2,1)
        return x,rot
 
 
