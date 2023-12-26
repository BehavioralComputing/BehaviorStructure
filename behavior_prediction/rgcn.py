import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.classify = nn.Linear(hid_feats, out_feats)
    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: self.classify(v) for k, v in h.items()}
        return h
    
device = torch.device('cuda:0')  
hetero_graph, _ = dgl.load_graphs("2_bhpre_Data_hetero_graph.dgl")

node_features = {}
hetero_graph = hetero_graph[0]
node_types = hetero_graph.ntypes

node_features = {}

node_features = node_features
user_id_nodes = hetero_graph.number_of_nodes('user_ID')
train_mask = torch.zeros(user_id_nodes, dtype=torch.bool)
train_mask[:] = True

hetero_graph.nodes['user_ID'].data['train_mask'] = train_mask
hetero_graph = hetero_graph.to(device)
labels = hetero_graph.nodes['user_ID'].data['label_user'].to(device)

for ntype in hetero_graph.ntypes:
    feat_shape = hetero_graph.nodes[ntype].data['feat'].shape
    print(f"Node type: {ntype}, Feature shape: {feat_shape}")

max_feat_dim = sum(hetero_graph.nodes[ntype].data['feat'].shape[1] for ntype in hetero_graph.ntypes)
start_idx = 0
for ntype in hetero_graph.ntypes:
    feat_dim = hetero_graph.nodes[ntype].data['feat'].shape[1]
    padding = torch.zeros((hetero_graph.number_of_nodes(ntype), max_feat_dim), dtype=torch.float32).to(device)
    padding[:, start_idx:start_idx+feat_dim] = hetero_graph.nodes[ntype].data['feat'].to(device)
    hetero_graph.nodes[ntype].data['feat'] = padding.to(device)
    start_idx += feat_dim

for node_type in node_types:
    node_type_feats = hetero_graph.nodes[node_type].data['feat']
    node_features[node_type] = node_type_feats.to(device)

model = RGCN(max_feat_dim, 64, 64, hetero_graph.etypes).to(device)



opt = torch.optim.Adam(model.parameters())

n_epoch = 100

for epoch in range(n_epoch):
    
    model.train()
    if epoch == n_epoch - 1:
        output_embeddings = model(hetero_graph, node_features)
        logits = output_embeddings['user_ID']
    else: 
        logits = model(hetero_graph, node_features)['user_ID']
    loss = F.cross_entropy(logits[train_mask], labels[train_mask].long())
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())
    if epoch == n_epoch-1:
        model.eval()
        with torch.no_grad():
            node_embeddings_conv1 = {}
            node_embeddings_liner = {}
            for key, value in model.conv1(hetero_graph, node_features).items():
                node_embeddings_conv1[key] = value.detach().cpu().numpy()
            for key, value in model(hetero_graph, node_features).items():
                node_embeddings_liner[key] = value.detach().cpu().numpy()
            with open('2_node_embeddings_liner.json', 'w') as file:
                json.dump({key: value.tolist() for key, value in node_embeddings_liner.items()}, file)

print('-------------------------------finish-----------------------')