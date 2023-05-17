from torch_sparse import SparseTensor

from torch_geometric.datasets import CitationFull, Planetoid, Twitch, Amazon, Coauthor
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import to_networkx, subgraph, degree
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv

import torch.nn.functional as F 
import torch.nn as nn
import torch

import networkx as nx
import numpy as np
from utils import parameter_parser


torch.manual_seed(10)
args = parameter_parser()




class MLPSamples:
    
    def __init__(self, data, num_inputs=10, num_targets=100):
        self.num_inputs = num_inputs
        self.num_targets = num_targets
        self.g = nx.Graph(data.edge_index.cpu().numpy().T.tolist())
        
    @classmethod
    def create_mapping(cls, ids): 
        return dict(zip(ids.cpu().numpy(), range(len(ids)), strict=True)), ids
    
    def create_component(self):
         for cc in sorted(nx.connected_components(self.g), key=len, reverse=True):
            self.cc = list(cc)     
            break
        
         return self.cc
        
    def sample(self, inputs= None, targets = None, inductive_eval=False):
        if inputs is None: # training
            size = self.num_targets + self.num_inputs
        else:
            size = self.num_targets 
            
        if inputs is None: # training
            
            if targets is None:
                # samples = torch.tensor(np.random.choice(self.cc, size=(size,), replace=False)) 
                # self.targets = samples[:self.num_targets]
                # self.inputs = samples[self.num_targets:]
                
                pr = nx.pagerank(self.g.subgraph(self.cc))
                samples = torch.tensor(sorted(pr.items(), key=lambda x:x[1], reverse=True))[:, 0].long()
                self.targets = samples[:self.num_targets]
                self.inputs = samples[self.num_targets : self.num_targets+self.num_inputs]
                
            else:
                inputs = torch.tensor(np.random.choice(self.cc, size=(size,), replace=False)) 
                self.inputs = torch.stack(list(set(inputs) - set(targets)))
                self.targets = targets
                
        else:
            if targets is None:
                pr = nx.pagerank(self.g.subgraph(self.cc))
                samples = torch.tensor(sorted(pr.items(), key=lambda x:x[1], reverse=True))[:, 0].long()
                self.targets = samples[:self.num_targets]
                #self.inputs = samples[self.num_targets : self.num_targets+self.num_inputs]
            else:
                self.targets= targets
            
            self.inputs = inputs
            
        if inputs is None:
            self.input2idx, self.id2input = self.create_mapping(self.inputs)
            self.target2idx, self.idx2target = self.create_mapping(self.targets)

            self._compute_input_to_target_distance()
        
        self._create_target_edges()
        self.inputs = self.inputs.to(args.device)
        self.targets = self.targets.to(args.device)
    
        
        
    def _compute_input_to_target_distance(self):
        self.shortest_path_lengths = torch.zeros(self.num_inputs, self.num_targets).to(args.device)       

        path_length = nx.shortest_paths.single_source_dijkstra_path_length
        self.shortest_path_lengths = torch.zeros(self.num_inputs, self.num_targets).to(args.device)
       
        for src_idx, source in enumerate(self.inputs):
            all_path_lens = path_length(self.g, source.item())
            for target, dist in all_path_lens.items():
                #print(source, target)
                if target in self.targets:
                    trg_idx = self.target2idx[target]
                    self.shortest_path_lengths[src_idx, trg_idx] = dist
        self.shortest_path_lengths = torch.sigmoid(self.shortest_path_lengths)
                    
            
    def _create_target_edges(self):
        target_edges = []
        for node in self.inputs:
            edges = zip([node] * self.targets.shape[0], self.targets, strict=True)
            target_edges.append(torch.tensor(list(edges)))
        self.target_edges = torch.cat(target_edges)  
