from torch_sparse import SparseTensor

from torch_geometric.datasets import CitationFull, Planetoid, Twitch, Amazon, Coauthor
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import to_networkx, subgraph, degree
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GAE, VGAE

import torch.nn.functional as F 
import torch.nn as nn
import torch

import networkx as nx
import numpy as np
from utils import parameter_parser


torch.manual_seed(10)
args = parameter_parser()


class connector_model(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(connector_model, self).__init__()
        self.lin1 = nn.Linear(input_dim, emb_dim)
        self.linear_mu = nn.Linear(emb_dim, emb_dim)
        self.linear_sigma = nn.Linear(emb_dim, emb_dim)
        self.lin2 = nn.Linear(emb_dim, emb_dim)
        
        self.bn = nn.BatchNorm1d(emb_dim)
        self.bn2 = nn.BatchNorm1d(emb_dim)
        self.bn3 = nn.BatchNorm1d(emb_dim)
        self.bn4 = nn.BatchNorm1d(emb_dim)
        
        
#         nn.init.xavier_uniform_(self.lin1.weight)
#         nn.init.xavier_uniform_(self.linear_mu.weight)
#         nn.init.xavier_uniform_(self.linear_sigma.weight)
#         nn.init.xavier_uniform_(self.lin2.weight)
            
    def forward(self, x):
        out = F.rrelu(self.lin1(x))
        out = self.bn(out)
        
#         mu =  F.rrelu(self.linear_mu(out))
#         mu = self.bn2(mu)
#         sigma = F.rrelu((self.linear_sigma(mu)))
#         sigma = self.bn3(sigma)
#         out = F.rrelu(self.lin2(sigma))
#         out = self.bn4(out)
        
#         mu =  F.rrelu(self.linear_mu(out))
#         mu = self.bn2(mu)
#         sigma = F.rrelu((self.linear_sigma(mu)))
#         sigma = self.bn3(sigma)
#         out = F.rrelu(self.lin2(sigma))
#         out = self.bn4(out)
        
        return out

    

    
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.conv_mu =GraphConv(in_channels, out_channels)
        self.conv_logstd = GraphConv(in_channels, out_channels)
        self.lin_mu = nn.Linear(out_channels, out_channels)
        self.lin_std = nn.Linear(out_channels, out_channels)
        self.lin2 = nn.Linear(out_channels, out_channels)
        self.lin3 = nn.Linear(out_channels, out_channels)
        
        #self.conv_mu =GCNConv(in_channels, out_channels)
        #self.conv_logstd = GCNConv(in_channels, out_channels)
      
    def forward(self, x, edge_index, edge_weight):
        
        if args.transductive:
            out = F.relu(self.conv(x, edge_index, edge_weight))
            mu = F.relu(self.lin_mu(((out))))
            log_std = F.relu(self.lin_std(((out))))
        else:
            #x = F.relu(self.conv(x, edge_index, edge_weight))
            mu = (self.conv_mu(x, edge_index, edge_weight))
            log_std = (self.conv_logstd(x, edge_index, edge_weight))
            
            mu = (self.lin_mu(mu))
            log_std = (self.lin_std(log_std)) 

        return mu, log_std
    

class Model(nn.Module):
    def __init__(self, input_dim, num_targets=100, output_dim=128, dropout=0.1):
        super().__init__()
        #input_dim = output_dim
        self.input_dim = input_dim
        self.num_targets = num_targets
        self.output_dim = output_dim
        self.dropout = dropout
        self.linear = connector_model(input_dim, num_targets)
        self.conv1 = VGAE(VariationalGCNEncoder(input_dim, output_dim))
        #self.conv2 = GraphConv(input_dim, output_dim)  
            
        self.conv2 = VGAE(VariationalGCNEncoder(input_dim, output_dim))
        self.conv3 = VGAE(VariationalGCNEncoder(output_dim, output_dim))
        self.conv4 = VGAE(VariationalGCNEncoder(output_dim, output_dim))
        
        
           
    def encode(self, x, edge_index, edge_weight=None, target = False):
        if edge_weight is None:
            x = self.conv2.encode(x, edge_index=edge_index)
            # x = (self.conv1.encode(x, edge_index=edge_index, edge_weight = torch.ones(edge_index.shape[1]).to(args.device)))
            # x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = F.rrelu(self.conv2.encode(x, edge_index=edge_index, edge_weight=edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # x = (self.conv3.encode(x, edge_index=edge_index, edge_weight=edge_weight))
            # x = F.dropout(x, p=self.dropout, training=self.training)

        return x 
    
        
        
    def forward(self, x, message_edge_index, target_edge_index=None, target_edge_weights = None, mlp_inputs=None):
        """
        x.shape = [N, D]
        message_edge_index = [2, E']
        target_edge_inde  x = [2, M * num_targets]
        mlp_inputs = [M]
        """
        # output = [M, sample_size]
        if target_edge_index is None:
            return self.encode(x, message_edge_index, torch.ones(message_edge_index.shape[1]).to(args.device)), target_edge_weights


        target_edge_index = torch.stack((target_edge_index[1], target_edge_index[0]), dim = 0)
        #target_edge_weights = torch.sigmoid(target_edge_weights)
        
        edge_index = torch.cat([message_edge_index, target_edge_index], dim = 1)
        edge_weight = torch.cat((torch.ones(message_edge_index.shape[1]).to(args.device), target_edge_weights))
                  
        if args.transductive:
            out = (self.encode(x, edge_index, edge_weight)) #+ self.encode(x, target_edge_index.long(), target_edge_weights)) 
        else:
            out = F.rrelu(self.encode(x, edge_index, edge_weight) + self.encode(x, target_edge_index.long(), target_edge_weights)) 

        return out, target_edge_weights

    
    
    def loss(self, data, z: torch.Tensor, variational=True):
        loss = self.conv2.recon_loss(z, data.edge_index) 
        if variational:
            loss = loss + (1 / data.num_nodes) * self.conv2.kl_loss()
            
        return loss
    
    

