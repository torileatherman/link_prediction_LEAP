from torch_geometric.nn.models import DeepGraphInfomax
from torch_geometric.nn import SAGEConv, GCNConv

import torch.nn.functional as F
import torch.nn as nn
import torch

import os.path as osp

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE, GDC
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, GCNConv, GINConv
from torch_scatter import scatter
from torch.nn import BatchNorm1d, ReLU, Sequential

import torch
import torch.nn.functional as F
import numpy as np
import math


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)



class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
    def forward(self, x, edge_index):
        #print(edge_index.shape)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim=128, device= 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        #self.encoder = Encoder(self.input_dim, output_dim)        
        self.conv1 = SAGEConv(self.input_dim, output_dim)
        self.conv2 = SAGEConv(output_dim, output_dim)
      
    
    def forward(self, data):
        x = self.conv1(data.x.to_dense(), data.edge_index)
        x = self.conv2(x, data.edge_index)
        #z = self.encoder(data.x.to_dense(), data.edge_index)
        return x
    
    
    def e_loss(self, data, x):
        r"""Computes the loss given positive and negative random walks."""
        edge_index = data.edge_label_index
        labels = data.edge_label.long()

        # Positive loss.
        EPS = 0.0000001
        src, trg = edge_index

        src_x = x[src][labels.bool()]
        trg_x = x[trg][labels.bool()]

        h_start = src_x
        h_rest = trg_x

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        src_x = x[src][~labels.bool()]
        trg_x = x[trg][~labels.bool()]

        h_start = src_x
        h_rest = trg_x

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        loss = pos_loss + neg_loss
        return loss

    
    def loss(self, data, z: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
       
        e_loss = self.e_loss(data, z)
        return e_loss
    
    
    def load(self, path="hom_model.pt"):
        self.load_state_dict(torch.load(path))

    def save(self, path="hom_model.pt"):
        torch.save(self.state_dict(), path)

        
    
