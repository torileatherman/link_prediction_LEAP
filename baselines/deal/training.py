#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn 
import time
from tqdm import tqdm
from model import *
from utils import *
from copy import copy

from torch_geometric.utils import subgraph
from torch_geometric.nn import Node2Vec
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid, DeezerEurope, Twitch, CitationFull
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
from torch_geometric.utils import structured_negative_sampling



# In[2]:


def evaluate(x, edge_index, labels):
    from sklearn.metrics import roc_auc_score, average_precision_score
    s, t = edge_index
    s_emb = x[s]
    t_emb = x[t]

    scores = s_emb.mul(t_emb).sum(dim=-1).cpu().numpy()
    auc = roc_auc_score(y_true=labels, y_score=scores)
    ap = average_precision_score(y_true=labels, y_score=scores)
    return auc, ap


# In[3]:


root = "/data/pakdd2023/"
name = "Citeseer"
print(name)
dataset = CitationFull(root, name)
data = dataset.data

inductive = True

if inductive:
    train_mask = torch.rand(data.num_nodes) < 0.5
    val_mask = ~train_mask

    train_data = copy(data)
    train_data.edge_index, _ = subgraph(train_mask, data.edge_index, relabel_nodes=True)
    train_data.x = data.x[train_mask]

    val_data = copy(data)
    val_data.edge_index, _ = subgraph(val_mask, data.edge_index, relabel_nodes=True)
    val_data.x = data.x[val_mask]

    #print(train_data)
    #print(val_data)

    # For teacher model, split the training graph data for transductive setting
    lsp_transform = RandomLinkSplit(num_val=0.0, num_test=0)
    train_data, _ , _ = lsp_transform(
        Data(
            x = train_data.x,
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes
        )
    )
    
    lsp_transform = RandomLinkSplit(num_val=0.0, num_test=0)
    valid_data, _, _ = lsp_transform(
        Data(
            x = val_data.x,
            edge_index=val_data.edge_index,
            num_nodes=val_data.num_nodes
        )
    )


else:
    lsp_transform = RandomLinkSplit()
    train_data, valid_data, test_data = lsp_transform(
        Data(
            x = data.x,
            edge_index=data.edge_index,
            num_nodes=data.num_nodes
        )
    )


# In[10]:


device = torch.device('cuda:1')
neg_num = 1

# node / attr /  inter
theta_list = (0.1,0.85,0.05)
lambda_list = (0.1,0.85,0.05)


# In[5]:


#torch.save((train_data, valid_data, test_data), "split.pt")
#train_data, valid_data, test_data = torch.load("split.pt")


# In[6]:


dist = precompute_dist_data(train_data.edge_index, num_nodes=train_data.num_nodes)


# In[7]:


deal = DEAL(64, train_data.x.shape[1], train_data.x.shape[0], device, None)


# In[13]:


optimizer2 = torch.optim.Adam(deal.parameters(), lr=5e-2)
epochs = 3000
losses = []
for epoch in range(epochs):
    deal.train()
    loss = deal.default_loss(train_data.edge_label_index.T, train_data.edge_label.to(device), torch.tensor(dist).to(device), train_data, thetas=theta_list, train_num=train_data.edge_label_index.T.shape[0])

    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    losses.append(loss.item())

    with torch.no_grad():
        deal.eval()
        
        _, x = deal.evaluate_inductive(valid_data.edge_label_index.T, valid_data, lambda_list)
        val_loss = deal.default_loss(valid_data.edge_label_index.T, valid_data.edge_label.to(device), torch.tensor(dist).to(device), valid_data, thetas=theta_list, train_num=valid_data.edge_index.T.shape[0])
    
        auc, ap = evaluate(
            x,
            valid_data.edge_label_index,
            valid_data.edge_label.long()
        )
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, "
          f"student loss: {loss.item():.8f}, "
          f"val loss: {val_loss.item():.8f}, "    
          f"validation AUC: {auc}, AP: {ap}")



# In[17]:


get_ipython().system('jupyter nbconvert --execute  training.ipynb')

