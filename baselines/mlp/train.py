import torch.nn.functional as F
import torch.nn as nn
import torch
import pickle
import os
from torch_geometric.datasets import Twitch, AttributedGraphDataset, CitationFull, WikipediaNetwork, Planetoid, Coauthor
from torch_geometric.transforms import GDC, RandomLinkSplit
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj, to_networkx

from copy import deepcopy, copy
from torch_geometric.utils import subgraph

torch.manual_seed(10)
device = "cuda:1"


def open_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data



def evaluate(x, data):
    from sklearn.metrics import roc_auc_score, average_precision_score
    edge_index = data.edge_label_index
    labels = data.edge_label.long().cpu()
    
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
name = "ACM"

inductive = False

if inductive:
    
    train_data = open_file('/home/aemad/PycharmProjects/LEAP/datasplits/'+'ind'+name+'_train_data.pickle')
    valid_data = open_file('/home/aemad/PycharmProjects/LEAP/datasplits/'+'ind'+name+'_valid_data.pickle')
    test_data = open_file('/home/aemad/PycharmProjects/LEAP/datasplits/'+'ind'+name+'_test_data.pickle')

else:
    train_data = open_file('/home/aemad/PycharmProjects/Graph2Feat/datasplits/'+name+'_train_data.pickle')
    valid_data = open_file('/home/aemad/PycharmProjects/Graph2Feat/datasplits/'+name+'_valid_data.pickle')
    test_data = open_file('/home/aemad/PycharmProjects/Graph2Feat/datasplits/'+name+'_test_data.pickle')


torch.manual_seed(10) 

    
class indStudent(nn.Module):
    def __init__(self, input_dim, emb_dim, device):
        super(indStudent, self).__init__()
        self.device = device
        self.Linear1 = nn.Linear(input_dim, emb_dim).to(self.device)
        self.Linear2 = nn.Linear(emb_dim, emb_dim).to(self.device)
        #self.prelu = nn.PReLU()
        
    def forward(self, x, z = None):
        x = F.rrelu(self.Linear1(x.to(device)))
        #x = F.rrelu(self.Linear2(x.to(device)))
        # if z is not None:
        #      x = F.rrelu(torch.add(x, z)/2)
        return x               
    
    
torch.manual_seed(10) 
# if inductive:
#     student = nn.Sequential(
#           nn.Linear(train_data.num_features, 5 * 128),
#           nn.RReLU(),
#           nn.Linear(5* 128, 128),
#           nn.RReLU()).to(device)

# else:
#     student = nn.Sequential(
#           nn.Linear(train_data.num_features, 5 * 128),
#           nn.RReLU(),
#           nn.Linear(5* 128, 128),
#           nn.RReLU()).to(device)


distill = False
#optimizer2 = torch.optim.Adam(student.parameters(), lr=0.001)
epochs =2000
losses = []
best_student = None
best = 0.0
beta = [0.005]
gamma = [1]
for b in beta:
    for q in gamma:
        
        if inductive:
            student = indStudent(train_data.num_features, 128, device).to(device)
        else:
            student = indStudent(train_data.num_features, 128, device).to(device)
       
        optimizer2 = torch.optim.Adam(student.parameters(), lr=0.001)
 
        best = 0.0
        best_student = None
        losses = []
   
        print(q, b)
    
        for epoch in range(epochs):
            torch.manual_seed(10)
            student.train()
            loss = 0
            student_x = student(train_data.x.to(device))
            if distill:
                loss += q * F.mse_loss(student_x, z.detach())   
            loss +=  b * e_loss(student_x, train_data)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            losses.append(loss.item())

            with torch.no_grad():
                student.eval()
                x = student(valid_data.x.to(device))
                val_loss = e_loss(x, valid_data)
                auc, ap = evaluate(x, valid_data)

            if ap > best :
                best = ap
                best_student = deepcopy(student)
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, "
                  f"student loss: {loss.item():.8f}, "
                  f"val loss: {val_loss.item():.8f}, "    
                  f"validation AUC: {auc}, AP: {ap}")

        import time

        # #testing       
        with torch.no_grad():
            best_student.eval()
            test_data = test_data.to(device)
            t_0 = time.time()
            student_x = best_student(test_data.x)    
            t_1 = time.time()
            elapsed_time = round((t_1 - t_0) * 10 ** 3, 3)
            print(elapsed_time)
            auc, ap = evaluate(student_x, test_data)

            print(f"testing AUC: {auc}, AP: {ap}")