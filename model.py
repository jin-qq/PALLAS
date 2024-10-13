from torch import distributions as dists
import torch.nn as nn
import torch
# from torch.nn import c
from torch.nn.functional import relu,leaky_relu
import math
from module_utils import *

from collections import OrderedDict
class Latend_Enc(nn.Module):
    def __init__(self,in_dim,in_channels,down_sample:bool, out_dim,latent_dim=128,out_channels=None, conv_shortcut=False,temb_channels=10):
        super(Latend_Enc,self).__init__()
        self.enc=ResnetBlock(in_dim,in_channels,down_sample,temb_channels)
        self.attn=AttnBlock(in_channels)
        self.proj=nn.Sequential(OrderedDict([
          ('conv1', nn.Linear(in_dim,256)),
          ('relu1', nn.LeakyReLU()),
          ('conv2', nn.Linear(256,out_dim)),
          ('relu2', nn.LeakyReLU())
        ]))
        self.channels=in_channels
    def forward(self,x,t):
        # temb=get_timestep_embedding(t, self.channels)
        return self.proj(self.attn(self.enc(x,temb).permute(0,2,1)))
    
class VAE(torch.nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(input_size, hidden_size, latent_size)
        self.decoder = VAE_Decoder(latent_size, hidden_size, output_size)
    def forward(self, x):
        mu,sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + eps*sigma
        re_x = self.decoder(z)
        return re_x,mu,sigma
        
        
        
        
        
        
         
