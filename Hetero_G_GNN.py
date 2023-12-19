

import torch
import numpy as np
import pandas as pd
df = pd.read_csv('Fraud.csv')
df.head()


# In[2]:


df = df[:100000]




from sklearn.preprocessing import LabelEncoder
encoder = {}
for i in df.select_dtypes('object').columns:
    encoder[i] = LabelEncoder()
    df[i] = encoder[i].fit_transform(df[i])




df["step"] = df["step"]-1
df["type"] = df["type"]-1
df




#分箱
df['amount'],cut_bin = pd.qcut(df['amount'].rank(method="first"), q=4, labels=[0, 1, 2, 3], retbins=True)
df['oldbalanceOrg'],cut_bin = pd.qcut(df['oldbalanceOrg'].rank(method="first"), q=4, labels=[0, 1, 2, 3], retbins=True)
df['newbalanceOrig'],cut_bin = pd.qcut(df['newbalanceOrig'].rank(method="first"), q=4, labels=[0, 1, 2, 3], retbins=True)
df['oldbalanceDest'],cut_bin = pd.qcut(df['oldbalanceDest'].rank(method="first"), q=4, labels=[0, 1, 2, 3], retbins=True)
df['newbalanceDest'],cut_bin = pd.qcut(df['newbalanceDest'].rank(method="first"), q=4, labels=[0, 1, 2, 3], retbins=True)



df["id"] = df.index


from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
data = HeteroData()



import torch
from torch import Tensor
unique_id = df["id"].unique()
unique_step = df["step"].unique()
unique_type = df["type"].unique()
unique_nameOrig = df["nameOrig"].unique()
unique_nameDest = df["nameDest"].unique()
data["step"].node_id = torch.arange(len(unique_step))
data["id"].node_id = torch.arange(len(unique_id))
data["type"].node_id = torch.arange(len(unique_type))
data["nameOrig"].node_id = torch.arange(len(unique_nameOrig))
data["nameDest"].node_id = torch.arange(len(unique_nameDest))




edge_index_id_to_step = torch.tensor([df["id"], df["step"]], dtype=torch.int64)
edge_index_id_to_type = torch.tensor([df["id"], df["type"]], dtype=torch.int64)
edge_index_id_to_nameOrig = torch.tensor([df["id"], df["nameOrig"]], dtype=torch.int64)
edge_index_id_to_nameDest = torch.tensor([df["id"], df["nameDest"]], dtype=torch.int64)
edge_index_nameOrig_to_nameDest = torch.tensor([df["nameOrig"], df["nameDest"]], dtype=torch.int64)




data["id", "step"].edge_index = edge_index_id_to_step
data["id", "type"].edge_index = edge_index_id_to_type
data["id", "nameOrig"].edge_index = edge_index_id_to_nameOrig
data["id", "nameDest"].edge_index = edge_index_id_to_nameDest
data["nameOrig", "nameDest"].edge_index = edge_index_id_to_nameDest





data.metadata()




data.edge_index_dict





data = T.ToUndirected()(data)





transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    is_undirected=True,
    add_negative_train_samples=False,
    edge_types=[("nameOrig", "to", "nameDest")],
    rev_edge_types=[("nameDest", "rev_to", "nameOrig")], 
)


train_data, val_data, test_data = transform(data)





from torch_geometric.loader import LinkNeighborLoader
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[2, 5],
    neg_sampling_ratio=2.0,
    edge_label_index=(("nameOrig", "to", "nameDest"), train_data["nameOrig", "to", "nameDest"].edge_label_index),
    edge_label=train_data["nameOrig", "to", "nameDest"].edge_label,
    batch_size=128,
    shuffle=True,
)




temp = 0
for i in train_loader:
    temp =temp + 1
    print(temp)




from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Define a 2-layer GNN computation graph.
        # Use a *single* `ReLU` non-linearity in-between.
        # TODO:
        #print(x, edge_index)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, nameOrig: Tensor, nameDest: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_nameOrig = nameOrig[edge_label_index[0]]
        edge_feat_nameDest = nameDest[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_nameOrig * edge_feat_nameDest).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        #self.fraud = torch.nn.Linear(20, hidden_channels)
        #self.nameOrig_emb = torch.nn.Embedding(128, hidden_channels)
        self.node_emb = torch.nn.Embedding(128, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        #self.gnn = to_hetero(self.gnn, node_types=node_types, edge_types=edge_types)
        self.classifier = Classifier()
    def forward(self, data: HeteroData) -> Tensor:
        #x_dict = {
        #  "nameOrig": self.node_emb(data["nameOrig"].node_id),
        #  "nameDest": self.node_emb(data["nameDest"].node_id),
        #} 
        x_dict = {} 
        for i in data.node_id_dict.keys():
            x_dict[i] = self.node_emb(data[i].node_id)
        print(x_dict)
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["nameOrig"],
            x_dict["nameDest"],
            data["nameOrig", "to", "nameDest"].edge_label_index,
        )
        return pred

        
model = Model(hidden_channels=64)




import tqdm
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 6):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        print(len(sampled_data))
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data["nameOrig", "to", "nameDest"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")





for epoch in range(1,6):
    train_iterator = iter(train_loader)  # 获取新的迭代器对象
    print(len(train_iterator))
    for batch in train_iterator:
        print(len(batch))




train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[30] * 2,
    edge_label_index=(("nameOrig", "to", "nameDest"), train_data["nameOrig", "to", "nameDest"].edge_label_index),
    #edge_label=train_data["nameOrig", "to", "nameDest"].edge_label,
    batch_size=128,
)
num_iterations = len(train_loader)  # 迭代次数

for i, data in enumerate(train_loader):
    # 执行训练步骤
    print(i+1)

print("Total iterations:", num_iterations)






