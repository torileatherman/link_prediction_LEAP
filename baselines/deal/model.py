import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import torch_geometric as tg
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import init

class Hidden_Layer(nn.Module): #Hidden Layer, Binary classification
        
    def __init__(self, emb_dim, device, BCE_mode, mode='all', dropout_p = 0.3):
        super(Hidden_Layer, self).__init__()
        self.emb_dim = emb_dim
        self.mode = mode
        self.device = device
        self.BCE_mode = BCE_mode
        self.Linear1 = nn.Linear(self.emb_dim*2, self.emb_dim).to(self.device)
        self.Linear2 = nn.Linear(self.emb_dim, 32).to(self.device)
        x_dim = 1
        self.Linear3 = nn.Linear(32, x_dim).to(self.device)
        if self.mode == 'all':
            if self.BCE_mode:
                self.linear_output = nn.Linear(x_dim+ 3, 1).to(self.device)
            else:
                self.linear_output = nn.Linear(x_dim+ 3, 2).to(self.device)
        else:
            self.linear_output = nn.Linear(1, 2).to(self.device) 
            self.linear_output.weight.data[1,:] = 1
            self.linear_output.weight.data[0,:] = -1

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.pdist = nn.PairwiseDistance(p=2,keepdim=True)       
        self.softmax = nn.Softmax(dim=1)
        self.elu = nn.ELU()
        assert (self.mode in ['all','cos','dot','pdist']),"Wrong mode type"


    def forward(self, f_embs, s_embs):

        if self.mode == 'all':
            x = torch.cat([f_embs,s_embs],dim=1)
            x = F.rrelu(self.Linear1(x))
            x = F.rrelu(self.Linear2(x))
            x = F.rrelu(self.Linear3(x))
            cos_x = self.cos(f_embs,s_embs).unsqueeze(1)
            dot_x = torch.mul(f_embs,s_embs).sum(dim=1,keepdim=True)
            pdist_x = self.pdist(f_embs,s_embs)
            x = torch.cat([x,cos_x,dot_x,pdist_x],dim=1)
        elif self.mode == 'cos':
            x = self.cos(f_embs,s_embs).unsqueeze(1)
        elif self.mode == 'dot':
            x = torch.mul(f_embs,s_embs).sum(dim=1,keepdim=True)
        elif self.mode == 'pdist':
            x = self.pdist(f_embs,s_embs)

        if self.BCE_mode:
            return x.squeeze()
            # return (x/x.max()).squeeze()
        else:
            x = self.linear_output(x)
            x = F.rrelu(x)
            # x = torch.cat((x,-x),dim=1)
            return x
    
    def evaluate(self, f_embs, s_embs):
        if self.mode == 'all':
            x = torch.cat([f_embs,s_embs],dim=1)
            x = F.rrelu(self.Linear1(x))
            x = F.rrelu(self.Linear2(x))
            x = F.rrelu(self.Linear3(x))
            cos_x = self.cos(f_embs,s_embs).unsqueeze(1)
            dot_x = torch.mul(f_embs,s_embs).sum(dim=1,keepdim=True)
            pdist_x = self.pdist(f_embs,s_embs)
            x = torch.cat([x,cos_x,dot_x,pdist_x],dim=1)
        elif self.mode == 'cos':
            x = self.cos(f_embs,s_embs)
        elif self.mode == 'dot':
            x = torch.mul(f_embs,s_embs).sum(dim=1)
        elif self.mode == 'pdist':
            x = -self.pdist(f_embs,s_embs).squeeze()
        return x
    
    
class Emb(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=False, layer_num=2, dropout=0, device='cpu', **kwargs):
        super(Emb, self).__init__()
        self.attr_emb = nn.Embedding(input_dim , output_dim)
        self.attr_num = input_dim
        self.device = device
        self.lin = nn.Linear(output_dim, output_dim)
        self.lin2 = nn.Linear(output_dim, output_dim)

    def forward(self, data):
        x = data.x.to(self.device)
        x = torch.mm(x, self.attr_emb(torch.arange(self.attr_num).to(self.attr_emb.weight.device)))
        x = self.lin(x)
        x= self.lin2(x)
        return x
    
    
class DEAL(nn.Module):

    def __init__(self, emb_dim, attr_num, node_num,device, args, num_classes=0 ,feature_dim=64,dropout_p = 0.3, verbose=False):
        super(DEAL, self).__init__()
        n_hidden= 2
        self.device = device
        self.node_num = node_num
        self.attr_num = attr_num
        self.emb_dim = emb_dim
        self.mode = 'cos'
        self.verbose = verbose
        self.BCE_mode = True
        self.gamma = 2

        self.num_classes = num_classes
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.pdist = nn.PairwiseDistance(p=2,keepdim=True)       
        self.softmax = nn.Softmax(dim=1)


        self.dropout = nn.Dropout(p=dropout_p)
        if self.BCE_mode:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        if self.num_classes:
            self.nc_Linear = nn.Linear(self.emb_dim,self.num_classes).to(self.device)
            nn.init.xavier_uniform_(self.nc_Linear.weight)
            
        self.nc_W = nn.Linear(2 * self.emb_dim,self.emb_dim).to(self.device)
        nn.init.xavier_uniform_(self.nc_W.weight)
        self.inter_W = nn.Linear(self.emb_dim,self.emb_dim, bias=False).to(self.device)

        self.node_emb = nn.Embedding(node_num, emb_dim).to(self.device)

        self.attr_emb = Emb(input_dim=attr_num, feature_dim= emb_dim,
                                hidden_dim=emb_dim, output_dim=emb_dim,
                                feature_pre=True, layer_num=0 if n_hidden is None else n_hidden,
                                dropout=dropout_p, device= device).to(device)

        self.node_layer = Hidden_Layer(self.emb_dim,self.device,self.BCE_mode, mode=self.mode)
        # self.attr_layer = self.node_layer
        # self.inter_layer = self.node_layer
        self.attr_layer = Hidden_Layer(self.emb_dim,self.device,self.BCE_mode, mode=self.mode)
        self.inter_layer = Hidden_Layer(self.emb_dim,self.device,self.BCE_mode, mode=self.mode)
        
    def node_forward(self, nodes):
        first_embs = self.node_emb(nodes[:,0])
        sec_embs = self.node_emb(nodes[:,1])
        return self.node_layer(first_embs,sec_embs)
    
    def attr_forward(self, nodes,data):
        node_emb = self.dropout(self.attr_emb(data))
        attr_res = self.attr_layer(node_emb[nodes[:,0]],node_emb[nodes[:,1]])
        return attr_res
    
    def inter_forward(self, nodes,data):
        first_nodes = nodes[:,0]
        first_embs = self.attr_emb(data)
        # first_embs = self.inter_W(first_embs)
        first_embs = self.dropout(first_embs)[first_nodes]
        sec_embs = self.node_emb(nodes[:,1])
        return self.inter_layer(first_embs,sec_embs)


    def RLL_loss(self,scores,dists,labels,alpha=0.2, mode='cos', gamma_1= 2, gamma_2 = 2):

        #gamma_1 = gamma1
        #gamma_2 = gamma2
        b_1 = 0.1
        b_2 = 0.1
        return torch.mean(labels*(torch.log(1+torch.exp(-scores*gamma_1+b_1)))/gamma_1+ torch.exp(dists)*(1-labels)*torch.log(1+torch.exp(scores*gamma_2+b_2))/gamma_2)

    def default_loss(self,inputs, labels, dists, data, thetas=(1,1,1), train_num = 1330,c_nodes=None, c_labels=None, gamma1=2, gamma2=2):
        if self.BCE_mode:
            labels = labels.float()
        nodes = inputs.to(self.device)
        labels = labels.to(self.device)
        dists = dists[nodes[:,0],nodes[:,1]] 

        loss_list = []

        scores = self.node_forward(nodes) 
        node_loss = self.RLL_loss(scores, dists,labels, gamma_1=gamma1, gamma_2=gamma2)
        loss_list.append(node_loss*thetas[0])

        scores = self.attr_forward(nodes,data)
        attr_loss = self.RLL_loss(scores,dists,labels, gamma_1=gamma1, gamma_2=gamma2)
        loss_list.append(attr_loss*thetas[1])
         
        unique_nodes = torch.unique(nodes)
        first_embs = self.attr_emb(data)[unique_nodes]

        sec_embs = self.node_emb(unique_nodes)
        loss_list.append(-self.cos(first_embs,sec_embs).mean()*thetas[2])

        losses = torch.stack(loss_list)
        self.losses = losses.data
        return losses.sum()

    def evaluate(self, nodes,data, lambdas=(1,1,1)):
       
        node_emb = self.node_emb(torch.arange(data.num_nodes).to(self.device)) 
        nodes = nodes.to(self.device)
        first_embs = node_emb[nodes[:,0]]
        sec_embs = node_emb[nodes[:,1]]
        res = self.node_layer(first_embs,sec_embs) * lambdas[0]

        f_emb = self.attr_emb(data)
        first_embs = f_emb[nodes[:,0]]
        sec_embs = f_emb[nodes[:,1]]
        res = res + self.attr_layer(first_embs,sec_embs)* lambdas[1]

        
        first_nodes = nodes[:,0]
        first_embs = self.attr_emb(data)[first_nodes]
        sec_embs = self.node_emb(nodes[:,1])
        res = res + self.inter_layer(first_embs,sec_embs)* lambdas[2]

        return res, torch.cat([f_emb,node_emb],dim=1)
        
        
    def evaluate_inductive(self, nodes,data, lambdas=(1,1,1)):
       
        f_emb = self.attr_emb(data)
        first_embs = f_emb[nodes[:,0]]
        sec_embs = f_emb[nodes[:,1]]
        res = self.attr_layer(first_embs,sec_embs)* lambdas[1]

        return res, f_emb
    