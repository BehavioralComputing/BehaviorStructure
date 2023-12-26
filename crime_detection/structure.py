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
import time
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)


class Aggregationfeature(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        # Linear layer to compress input features from in_dim to hidden_dim
        self.input_linear = nn.Linear(in_dim, hidden_dim)
        self.rgcn = RGCN(hidden_dim, hidden_dim, hidden_dim, rel_names)
        self.mlp = MLP(hidden_dim, hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feat']
        h = {k: F.relu(self.input_linear(v)) for k, v in h.items()}
        
        h = self.rgcn(g, h)
        
        with g.local_scope():
            g.ndata['h'] = h
            all_mean_feat = torch.zeros(len(data), hid_num).to(device)
            for index, row in data.iterrows():
                feat_sum = torch.zeros(hid_num).to(device)  
                for feature in features:
                    num = 0
                    if pd.notna(row[feature]):
                        num = num + 1
                        if feature in feat_dict:       
                            feat_sum += h[feature][feat_dict[feature][str(row[feature])]]
                mean_feat = feat_sum / num   
                all_mean_feat[index, :] = mean_feat     
            return self.mlp(all_mean_feat)

device = torch.device('cuda:0')  
loaded_graphs, _ = dgl.load_graphs("Crime_Data_hetero_graph.dgl")

features = ["AREA NAME", "Rpt Dist No", "Part 1-2", "Vict Age", "Vict Sex", "Vict Descent", "Premis Desc","Weapon Desc","Status", 
                      "Cross Street", "Month_Rptd", "Day_Rptd", "Year_Rptd", "Month_OCC", "Day_OCC", "Year_OCC", "Date Difference", "Hour", "Minute"]
label = 'Label'

data = pd.read_csv('./data/Pre-processing_Crime_Data_Top10.csv', dtype={'Vict Age': float})
dup_data = data

with open("column_encoding.json", "r") as file:
    feat_dict = json.load(file)

loaded_graphs = loaded_graphs[0]
max_feat_dim = 768

g = loaded_graphs.to(device)

etypes = g.etypes
hid_num = 128
def evaluate(g,  labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

learning_rate = 0.001

all_reports = ""
for i in range(10):
    
    data = data.sample(n=20000, random_state=i).reset_index(drop=True)
    X = data[features]
    y = torch.tensor(data['Label'].values, dtype=torch.long).to(device)
    train_val_test_idx = np.arange(len(data))
    train_idx, test_idx = train_test_split(train_val_test_idx, test_size=0.5, random_state=i)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=i)
    train_mask = torch.zeros(len(data), dtype=torch.bool)
    val_mask = torch.zeros(len(data), dtype=torch.bool)
    test_mask = torch.zeros(len(data), dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    model = Aggregationfeature(max_feat_dim, hid_num, 10, etypes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    for epoch in range(300):
        model.train()
        epoch_loss = 0
        labels = y.to(device)
        logits= model(g)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask].squeeze(-1))  
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = evaluate(g,  labels, val_mask, model)
        
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), 'strcture_best_model.pt')


        print(
                "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                    epoch, loss.item(), acc
                )
            )
        
    model.load_state_dict(torch.load('strcture_best_model.pt'))
    model.eval()
    test_predictions = []
    test_labels = []

    labels = labels.to(device)

    with torch.no_grad():
        logits = model(g)
        predictions = torch.argmax(logits[test_mask], dim=1)

        test_predictions.extend(predictions.tolist())
        test_labels.extend(labels[test_mask].tolist())
        report = classification_report(test_labels, test_predictions,  digits=5)
        print("Classification Report:\n", report)
        all_reports = f"RCGN_Classification Report hid_num {hid_num} round {i} layers_2:\n{report}\n\n"
        with open("./report/strcture_classification_reports.txt", "a") as report_file:
            report_file.write(all_reports)


