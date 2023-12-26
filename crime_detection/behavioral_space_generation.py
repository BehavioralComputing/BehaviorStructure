import pandas as pd
import dgl
import torch
import math
import numpy as np
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertModel

def del_nan(A,B):
    array_A = np.array(A)
    array_B = np.array(B)
    nan_mask = np.isnan(array_A) | np.isnan(array_B)
    valid_indices = ~nan_mask
    return array_A[valid_indices], array_B[valid_indices]

def complete_graph_edge(data_dict,columns_to_convert, data_list):

    for src in columns_to_convert:
        for tar in columns_to_convert:
            if src != tar:
                new_list_A, new_list_B = del_nan(data_list[src], data_list[tar])
                data_dict.update({
                    (src, 'edge_' + src + '_' + tar, tar): (torch.tensor(new_list_A, dtype=torch.int),
                                                torch.tensor(new_list_B, dtype=torch.int))
            }) 

device = torch.device('cuda:1')
data_df = pd.read_csv("./data/Pre-processing_Crime_Data_Top10.csv")

columns_to_convert = ["AREA NAME", "Rpt Dist No", "Part 1-2", "Vict Age", "Vict Sex", "Vict Descent", "Premis Desc","Weapon Desc","Status", 
                      "Cross Street", "Month_Rptd", "Day_Rptd", "Year_Rptd", "Month_OCC", "Day_OCC", "Year_OCC", "Date Difference", "Hour", "Minute"]


column_encoding = {}
feature_lists = {}
data_list = {}

for col in columns_to_convert:
    unique_values = data_df[col].dropna().unique()
    encoding = {value: i for i, value in enumerate(unique_values)}
    feature_lists[col] = [value for value in unique_values]
    column_encoding[col] = encoding

for col, encoding in column_encoding.items():
    data_df[col] = data_df[col].map(encoding)

data_list = {column: data_df[column].tolist() for column in columns_to_convert}

data_dict={}

tim_pla_peo_weap = ["Vict Age", "Vict Sex", "Vict Descent", "Premis Desc","Weapon Desc",
                       "Month_OCC", "Day_OCC", "Year_OCC"   , "Hour", "Minute"]

pcl = ["Premis Desc", "Cross Street"]

rpa = ["Rpt Dist No", "Premis Desc", "AREA NAME"]

rsp = ["Status", "Part 1-2", "Rpt Dist No"]

rdp = ["Day_Rptd", "Rpt Dist No", "Premis Desc"]

dym = ["Month_Rptd", "Day_Rptd", "Year_Rptd"]

ddd = ["Date Difference", "Day_Rptd", "Day_OCC"]

graph_all = [tim_pla_peo_weap, pcl, rpa, rsp, rdp, dym, ddd]

for graph in graph_all:
    complete_graph_edge(data_dict, graph, data_list)



hetero_graph = dgl.heterograph(data_dict).to(device)


for column in columns_to_convert:
    
    feature_list = feature_lists[column]  
    feature_list_strings = list(map(str, feature_list))
    model_name = "bert-base-uncased"  
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    semantic_vectors = []
    for value in feature_list_strings:
        encoded_text = tokenizer(value, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded_text)
            semantic_vector = outputs.last_hidden_state.mean(dim=1).to(device) 
            semantic_vectors.append(semantic_vector)
    hetero_graph.nodes[column].data['feat'] = torch.cat(semantic_vectors, dim=0).to(device)
    
dgl.save_graphs("Crime_Data_hetero_graph.dgl", [hetero_graph])

