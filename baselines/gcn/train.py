import torch.nn.functional as F
import torch.nn as nn
import torch
import pickle
import os
import torch_geometric
from torch_geometric.datasets import Twitch, AttributedGraphDataset, CitationFull, WikipediaNetwork
from torch_geometric.transforms import GDC, RandomLinkSplit
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj, to_networkx

from copy import copy
from torch_geometric.utils import subgraph

torch.manual_seed(10)
device = "cuda:1"

    
def save_file(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    
def open_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def evaluate(x, data):
    from sklearn.metrics import roc_auc_score, average_precision_score
    edge_index = data.edge_label_index
    labels = data.edge_label.long()
    
    s, t = edge_index
    s_emb = x[s].detach().cpu()
    t_emb = x[t].detach().cpu()

    scores = s_emb.mul(t_emb).sum(dim=-1)
    auc = roc_auc_score(y_true=labels, y_score=scores)
    ap = average_precision_score(y_true=labels, y_score=scores)
    return auc, ap


def e_loss(x, data):
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




#train_data, valid_data, test_data = torch.load("split.pt")
root = "./data/pakdd2023/"
name = "CS"
print(name)

if os.path.isfile('/home/aemad/PycharmProjects/project_slkd/datasplits/'+name+'_train_data.pickle'):

    train_data = open_file('/home/aemad/PycharmProjects/project_slkd/datasplits/'+name+'_train_data.pickle')
    valid_data = open_file('/home/aemad/PycharmProjects/project_slkd/datasplits/'+name+'_valid_data.pickle')
    test_data = open_file('/home/aemad/PycharmProjects/project_slkd/datasplits/'+name+'_valid_data.pickle')


from GCN import GCN
from torch.quantization import quantize_dynamic
    


import time
test_data = test_data.to(device)
from torch.quantization import quantize_dynamic

model = GCN(input_dim=train_data.num_features, device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, torch_geometric.nn.SAGEConv}, dtype=torch.qint8
).to(device)

import torch.nn.utils.prune as prune

parameters_to_prune = (
    (model.conv1.lin_l, 'weight'),
    #(model.conv1.lin_r, 'weight'),
    (model.conv2.lin_l, 'weight'),
    #(model.conv2.lin_r, 'weight'),
    )
from torch_geometric.loader import NeighborLoader
test_loader = NeighborLoader(test_data, num_neighbors=[15]*1, batch_size=len(test_data), input_nodes= torch.arange(test_data.num_nodes))

model_acc = []
q_model_acc = []
s_model_acc = []
p_model_acc = []

for _ in range(1):
    
    model = GCN(input_dim=train_data.num_features, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, torch_geometric.nn.SAGEConv}, dtype=torch.qint8
    ).to(device)

    
    epochs = 1000
    best = 0.0

    for epoch in range(epochs):

        z = model(train_data.to(device))  
        loss = model.loss(train_data, z)  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()

        z = model(train_data)
        auc, ap = evaluate(z, valid_data)

        if (epoch + 1) % 10 == 0:
             print(f"Epoch {epoch + 1}/{epochs}, "
               f"Training loss: {loss:.4f}, "
               f"AUC: {auc}, AP: {ap}")

        if ap > best:
            best = ap    
            model.save(name+'_hom_model.pt')

        model.eval()


    t_0 = time.time()
    z = model(test_data)
    t_1 = time.time()
    model_acc.append(round((t_1 - t_0) * 10 ** 3, 3))


    t_0 = time.time()
    z = quantized_model(test_data)
    t_1 = time.time()
    q_model_acc.append(round((t_1 - t_0) * 10 ** 3, 3))
    
    
    sampled_data = next(iter(test_loader))
    t_0 = time.time()
    out = model(sampled_data)
    t_1 = time.time()
    s_model_acc.append(round((t_1 - t_0) * 10 ** 3, 3))
    

    prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5)
    
    t_0 = time.time()
    z = model(test_data)
    t_1 = time.time()
    p_model_acc.append(round((t_1 - t_0) * 10 ** 3, 3))
    
    

    
print(torch.mean(torch.tensor(model_acc)), torch.std(torch.tensor(model_acc)))
print(torch.mean(torch.tensor(q_model_acc)), torch.std(torch.tensor(q_model_acc)))
print(torch.mean(torch.tensor(p_model_acc)), torch.std(torch.tensor(p_model_acc)))
print(torch.mean(torch.tensor(s_model_acc)), torch.std(torch.tensor(q_model_acc)))
    
# auc, ap = evaluate(z, test_data)
# print(f"test AUC: {auc}, AP: {ap}")
