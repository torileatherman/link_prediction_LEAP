import torch.nn.functional as F
import torch.nn as nn
import torch
import pickle
import os
from tqdm import tqdm
from torch_geometric.datasets import Twitch, AttributedGraphDataset, CitationFull, WikipediaNetwork, Planetoid, Coauthor
from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj, to_networkx
from models.Model import *
from models.MLP import MLPSamples
from utils import parameter_parser
from torch_geometric.nn import DeepGraphInfomax, SAGEConv
from torch_geometric.transforms import RandomNodeSplit

from copy import deepcopy, copy
from torch_geometric.utils import subgraph

torch.manual_seed(10)
args = parameter_parser()

    
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
    labels =  labels
    

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

 
    
def get_scores(model, src, trg):
    r"""Computes the loss given positive and negative random walks."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    src_x = model(src)
    trg_x = model(trg)
    return src_x.mul(trg_x).sum(dim=-1)
    


#train_data, valid_data, test_data = torch.load("split.pt")
root = "./data/pakdd2023/"
if args.name == "Wikipedia":
    data = WikipediaNetwork(root+args.name, "chameleon").data
elif args.name == "crocodile":
    data = WikipediaNetwork(root+args.name, "crocodile", geom_gcn_preprocess=False).data
elif args.name == "PubMed":
    data = Planetoid(root+args.name, "PubMed").data
elif args.name == "Twitch":
    data = Twitch(root+args.name, "EN").data

print(args.name)


##### preparing and loading the data

if args.transductive:
    train_data = open_file('datasplits/'+args.name+'_train_data.pickle')
    valid_data = open_file('datasplits/'+args.name+'_valid_data.pickle')
    test_data = open_file('datasplits/'+args.name+'_test_data.pickle')
    
    train_data.nodes = torch.arange(train_data.num_nodes)
    valid_data.nodes = torch.arange(valid_data.num_nodes)
    test_data.nodes = torch.arange(test_data.num_nodes)

    
else:
    
    if os.path.isfile('datasplits/'+'ind'+args.name+'_train_data.pickle'):
        train_data = open_file('datasplits/'+'ind'+args.name+'_train_data.pickle')
        valid_data = open_file('datasplits/'+'ind'+args.name+'_valid_data.pickle')
        test_data = open_file('datasplits/'+'ind'+args.name+'_test_data.pickle')
    
    else:
        rnp = RandomNodeSplit(num_val=0.1, num_test=0.1)
        data = rnp(data.clone())

        adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.shape[1]), (data.num_nodes, data.num_nodes)).to_dense()
        train_edges = torch.zeros(data.num_nodes, data.num_nodes)
        val_edges = torch.zeros(data.num_nodes, data.num_nodes)
        test_edges = torch.zeros(data.num_nodes, data.num_nodes)
        for i in tqdm(data.train_mask.nonzero().view(-1)):
             for j in data.train_mask.nonzero().view(-1):
                    train_edges[i][j] = adj[i][j]             
                    
             for j in data.val_mask.nonzero().view(-1):
                    val_edges[i][j] = adj[i][j]             
        
             for j in data.test_mask.nonzero().view(-1):
                    test_edges[i][j] = adj[i][j]             
        
                
        train_data = Data()
        train_node_id = data.train_mask.nonzero().view(-1)
        train_data.x = data.x
        train_data.nodes = train_node_id #torch.arange(len(train_node_ind))
        train_data.edge_index = train_edges.nonzero().T  
        
        valid_data = Data()
        node_ind = data.val_mask.nonzero().view(-1)
        valid_data.x =data.x
        valid_data.nodes = node_ind 
        valid_data.edge_index = val_edges.nonzero().T
           
        test_data = Data()
        node_ind = data.test_mask.nonzero().view(-1)
        test_data.x = data.x #torch.cat((data.x[train_node_ind], data.x[node_ind]), dim = 0) 
        test_data.nodes = node_ind #torch.arange(len(test_data.x))[-len(node_ind):]
        test_data.edge_index = test_edges.nonzero().T
       
        rlp = RandomLinkSplit(num_val=0.0, num_test=0.0)
        train_data, _ , _ = rlp(train_data)
        valid_data, _ , _ = rlp(valid_data)
        test_data, _ , _ = rlp(test_data)

        save_file(train_data, 'datasplits/'+'ind'+args.name+'_train_data.pickle')
        save_file(valid_data, 'datasplits/'+'ind'+args.name+'_valid_data.pickle')
        save_file(test_data, 'datasplits/'+'ind'+args.name+'_test_data.pickle')
        


######### sampling anchors and retrieve the new nodes
num_targets = 250
num_inputs = 250


if num_targets > 0 and num_inputs > 0:
    
    mlp_samples = MLPSamples(train_data, num_inputs=num_inputs, num_targets=num_targets)
    cc = mlp_samples.create_component()
    mlp_samples.sample()        
    msg_edge_index = torch.tensor([ (u, v) for u, v in train_data.edge_index.T  
                                   if u not in set(mlp_samples.inputs) and v not in set(mlp_samples.inputs)]).T
    target_weight = 1 / (mlp_samples.shortest_path_lengths)

    if args.transductive:
        mlp_samples_eval = MLPSamples(train_data, num_inputs=num_inputs, num_targets=num_targets)
        mlp_samples_eval.cc = cc
        mlp_samples_eval.sample(inputs = mlp_samples.inputs, targets = mlp_samples.targets)                
        
        mlp_samples_test = MLPSamples(train_data, num_inputs=num_inputs, num_targets=num_targets)
        mlp_samples_test.cc = cc
        mlp_samples_test.sample(inputs = mlp_samples.inputs, targets = mlp_samples.targets)               

    else:
        mlp_samples_eval = MLPSamples(train_data, num_inputs=valid_data.num_nodes, num_targets=num_targets)
        mlp_samples_eval.cc = cc
        mlp_samples_eval.sample(inputs = valid_data.nodes, targets = mlp_samples.targets)                

        mlp_samples_test = MLPSamples(train_data, num_inputs=test_data.num_nodes, num_targets=num_targets)
        mlp_samples_test.cc = cc
        mlp_samples_test.sample(inputs = test_data.nodes, targets = mlp_samples.targets)               

    anchors = mlp_samples.targets
        
train_data = train_data.to(args.device)
valid_data = valid_data.to(args.device)
test_data = test_data.to(args.device)
###################################
    

## training leap    
mse_loss = nn.MSELoss()
cce = nn.CrossEntropyLoss()
connector_model = connector_model(train_data.num_features, 128).to(args.device)
model = Model(input_dim=train_data.num_features, num_targets=num_targets, dropout=0.99).to(args.device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)

   
best = 0.0
best_model = None
best_connector = None
best_epoch = 0

for epoch in range(args.epochs):    

    model.train()
    connector_model.train()
    
    # mlp_samples = MLPSamples(train_data, num_inputs=num_inputs, num_targets=num_targets)
    # cc = mlp_samples.create_component()
    # mlp_samples.sample(targets = anchors)        
    # msg_edge_index = torch.tensor([ (u, v) for u, v in train_data.edge_index.T  
    #                                if u not in set(mlp_samples.inputs) and v not in set(mlp_samples.inputs)]).T

    
    if num_targets > 0 and num_inputs > 0:
        
        target_edge_index = mlp_samples.target_edges.T.to(args.device).long()
        src_x = train_data.x[target_edge_index.T[:, 0]]
        trg_x = train_data.x[target_edge_index.T[:, 1]]

        target_edge_weights = get_scores(connector_model, src_x, trg_x)

        x, pred = model(x = train_data.x, 
                    message_edge_index = msg_edge_index.to(args.device), 
                    target_edge_index = target_edge_index,
                    target_edge_weights = target_edge_weights,
                    mlp_inputs=mlp_samples.inputs.to(args.device))
    
    else:
        target_edge_weights = torch.tensor([])
        x, pred = model(x = train_data.x, message_edge_index = train_data.edge_index.to(args.device))
        
        
    loss =  model.loss(train_data, x)  
    #loss =  e_loss(x, train_data)
    if num_targets > 0 and num_inputs > 0:
        loss +=  0.0005 * cce(target_edge_weights.flatten(), target_weight.flatten())

            
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    # validation
    with torch.no_grad():
        model.eval()
        connector_model.eval()
            
        if num_targets > 0 and num_inputs > 0:
            
            target_edge_index = mlp_samples_eval.target_edges.T.to(args.device).long()
            if target_edge_index.shape[0] != 0:
                src_x = valid_data.x[target_edge_index.T[:, 0]]
                trg_x = train_data.x[target_edge_index.T[:, 1]]
                target_edge_weights = get_scores(connector_model, src_x, trg_x)
            else:
                target_edge_weights = torch.tensor([])

            out, pred = model(x=valid_data.x,
                            message_edge_index=msg_edge_index.to(args.device),#.edge_index,
                            target_edge_index=target_edge_index,
                            target_edge_weights = target_edge_weights,
                            mlp_inputs=mlp_samples_eval.inputs.to(args.device))


        else:
            out, pred = model(x=valid_data.x, message_edge_index=train_data.edge_index.to(args.device))

        
        auc, ap = evaluate(out, valid_data)
        print(f"Epoch {epoch + 1}/{args.epochs}, Training loss: {loss.item():.10}, Validation AUC: {auc}, AP: {ap}")
        
        if ap > best:
            best = ap
            best_model = deepcopy(model)
            best_connector = deepcopy(connector_model)
            

#####################   
# testing 

with torch.no_grad():
        best_model.eval()
        best_connector.eval()
        
        if num_targets > 0 and num_inputs > 0:     
                
            target_edge_index = mlp_samples_test.target_edges.T.to(args.device).long()
            if target_edge_index.shape[0] != 0:
                src_x = test_data.x[target_edge_index.T[:, 0]]
                trg_x = train_data.x[target_edge_index.T[:, 1]]
                target_edge_weights = get_scores(best_connector, src_x, trg_x)
            else:
                target_edge_weights = torch.tensor([])

            
            out, pred = best_model(x=test_data.x,
                            message_edge_index=msg_edge_index.to(args.device),
                            target_edge_index=target_edge_index,
                            target_edge_weights = target_edge_weights,
                            mlp_inputs=mlp_samples_test.inputs.to(args.device))

        else:
            out, pred = best_model(x=test_data.x, message_edge_index=train_data.edge_index.to(args.device))
            
        auc, ap = evaluate(out, test_data)
        print(f"Epoch {epoch + 1}/{args.epochs}, Training loss: {loss.item():.10}, testing AUC: {auc}, AP: {ap}")
    
                      