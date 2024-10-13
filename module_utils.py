from torch import distributions as dists
import torch.nn as nn
import torch
from torch.nn import LeakyReLU
from torch.nn.functional import relu,leaky_relu
import math
# import math_ang as ang
from math import *
from torch.utils.data import *
import torch.nn.functional as F
from pos_embed import *

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def Iattention(q,k,v=None):
    if q.shape[1]!=q.shape[2]:
         raise ValueError("shape is not equal")
    if v==None:
        v=k
    qk=torch.matmul(q,k)/q.shape[0]
    return qk,torch.matmul(qk,v)
# calculate attention
def attention(q,k,v=None):
    if v==None:
        v=k
    qk=torch.matmul(q,k.transpose(-1,-2))/q.shape[0]
    return qk,torch.matmul(qk,v)

def reverse_attention(attn,qk):
    return torch.matmul(torch.inverse(qk), attn)
# reverse attention value

def reverse_cross_attention(attn,qk):
    v=reverse_attention(attn,qk)
    q=torch.matmul(qk*qk.shape[0],torch.inverse(v))
    return q,v
# reverse cross attention value

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False
            
class Coupling(nn.Module):

    def __init__(self, input_dim=3, mid_channels=256, num_layers=5):
        super().__init__()
        self.input_dim = input_dim
        self.mid_channels = mid_channels
        self.num_layers = num_layers

        #  scale and translation transforms
        self.s = nn.Sequential(*self._sequential(), nn.Tanh())
        self.t = nn.Sequential(*self._sequential())

    def _sequential(self):
        input_dim, mid_channels, num_layers = self.input_dim, self.mid_channels, self.num_layers
        sequence = [nn.Linear(input_dim, mid_channels), nn.ReLU()]  # first layer
        for _ in range(num_layers - 2):  # intermediate layers
            sequence.extend([nn.Linear(mid_channels, mid_channels), nn.ReLU()])
        sequence.extend([nn.Linear(mid_channels, input_dim)])  # final layer
        return sequence

    def forward(self, x):
        return self.s(x), self.t(x)


class RealNVP(nn.Module):
    def __init__(self, num_coupling_layers,input_dim,num_layers,middle_layer=512):
        super().__init__()
        if num_coupling_layers<=1:
            raise ValueError("num layers must larger than 1")
        self.num_coupling_layers = num_coupling_layers

        # model the latent as a
        self.distribution = dists.MultivariateNormal(loc=torch.zeros(2),
                                                     covariance_matrix=torch.eye(2))
        dig=torch.eye(input_dim).tolist()
        self.masks = torch.tensor(
           dig * (num_coupling_layers // 2), dtype=torch.float32
        )

        # create num_coupling_layers layers in the RealNVP network
        self.layers_list = [Coupling(input_dim=input_dim,num_layers=num_layers,mid_channels=middle_layer) for _ in range(num_coupling_layers)]

    def forward(self, x, do_forward=True):
        log_det_inv = 0.
        direction = 1
        if do_forward:
            direction = -1

        # pass through each coupling layer (optionally in reverse)
        for i in range(self.num_coupling_layers)[::direction]:
            mask =  self.masks[i]
            x_masked = x * mask
            reversed_mask = 1. - mask
            s, t = self.layers_list[i](x_masked)
            s = s * reversed_mask
            t = t * reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * torch.exp(direction * s) + direction * t * torch.exp(gate * s))
                + x_masked
            )
            # log det (and its inverse) are easily computed
            log_det_inv = log_det_inv + gate * s.sum(1)
        return x, log_det_inv

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -torch.mean(log_likelihood)

class IMHA_layer(nn.Module):
    def __init__(self,in_dim,num_layer=2,num_lin=5):
        super(IMHA_layer,self).__init__()
        if num_lin<=2:
            raise ValueError("num_linear must larger than 2")
        self.wq=RealNVP(num_layer,in_dim,num_lin)
        self.wk=RealNVP(num_layer,in_dim,num_lin)
        self.wv=RealNVP(num_layer,in_dim,num_lin)
        
    def forward(self,x,qk=None,do_forward=True):
        if do_forward==True:
            self.wq.to(x.device)
            q,_=self.wq(x,do_forward)
            self.wk.to(x.device)
            k,_=self.wk(x,do_forward)
            self.wv.to(x.device)
            v,_=self.wv(x,do_forward)
            qk,att= Iattention(q,k,v)
            return qk,att
        else:
            v=self.wv(reverse_attention(x,qk),inverse)
            return v,qk
        
class MHA_layer(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(MHA_layer,self).__init__()
        self.wq=torch.nn.Linear(in_dim,out_dim)
        self.wk=torch.nn.Linear(in_dim,out_dim)
        self.wv=torch.nn.Linear(in_dim,out_dim)
        
    def forward(self,x,freqs_cis):
        self.wq.to(x.device)
        q=self.wq(x)
        self.wk.to(x.device)
        k=self.wk(x)
        self.wv.to(x.device)
        v=self.wv(x)
        q,k=apply_rotary_emb(q,k,freqs_cis)
        qk,att= attention(q.flatten(2),k.flatten(2),v)
        
        return att

class ICA_layer(nn.Module):
    def __init__(self,in_dim,num_layer=3,num_lin=5):
        super(ICA_layer,self).__init__()
        if num_lin<=2:
            raise ValueError("num_linear must larger than 2")
        self.wq=RealNVP(num_layer,in_dim,num_lin)
        self.wk=RealNVP(num_layer,in_dim,num_lin)
        
        # self.projection
    def forward(self,x,qk=None,inverse=False):
        if inverse==False:
            self.wq.to(x.device)
            q=self.wq(x,inverse)
            self.wk.to(x.device)
            k=self.wk(x,inverse)
            return attention(q,k)
        else:
            q,k=reverse_cross_attention(inputer,qk)
            q=self.wq(q,inverse)
            k=self.wk(k,inverse)
            return q,k



# def nonlinearity(x):
#     # swish
#     return x*torch.sigmoid(x)


# def Normalize(in_channels):
#     return torch.nn.BatchNorm1d(num_features=in_channels)       


class Res_attention(nn.Module):
    def __init__(self,in_dim,hidden_dim,n_heads,freqs_cis):
        super(Res_attention,self).__init__()
        self.layer=MHA_layer(in_dim,hidden_dim*n_heads)
        self.norm=RMSNorm(hidden_dim*n_heads)
        self.act=nn.SiLU()
        self.freqs_cis=freqs_cis
    def forward(self,x):
        # return x+self.norm(self.act(self.layer(x,self.freqs_cis)))
        # x=self.norm(x)
        # x=
        return x+self.act(self.norm(self.layer(x,self.freqs_cis)))
