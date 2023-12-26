import pandas as pd
import dgl
import torch
import math
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertModel
import json

def del_nan(A,B):
    array_A = np.array(A)
    array_B = np.array(B)
    nan_mask = np.isnan(array_A) | np.isnan(array_B)
    valid_indices = ~nan_mask
    return array_A[valid_indices], array_B[valid_indices]

def star_graph_edge(data_dict, center, columns_to_convert, data_list):
    src = center
    for tar in columns_to_convert:
        if src != tar:
            new_list_A, new_list_B = del_nan(data_list[src], data_list[tar])
            data_dict.update({
                (src, 'edge_' + src + '_' + tar, tar): (torch.tensor(new_list_A, dtype=torch.int),
                                            torch.tensor(new_list_B, dtype=torch.int))
        }) 
            data_dict.update({
                (tar, 'edge_' + tar + '_' + src, src): (torch.tensor(new_list_B, dtype=torch.int),
                                            torch.tensor(new_list_A, dtype=torch.int))
        }) 

def complete_graph_edge(data_dict, columns_to_convert, data_list):

    for src in columns_to_convert:
        for tar in columns_to_convert:
            if src != tar:
                new_list_A, new_list_B = del_nan(data_list[src], data_list[tar])
                data_dict.update({
                    (src, 'edge_' + src + '_' + tar, tar): (torch.tensor(new_list_A, dtype=torch.int),
                                                torch.tensor(new_list_B, dtype=torch.int))
            }) 

def click_answer(data_df):
    data = data_df.loc[data_df['click_timestamp'] != 0].copy()
    return data
    


def get_unique_elements(X, Y):
    unique_elements = np.unique(np.concatenate([X, Y]))
    return unique_elements

device = torch.device('cuda:0')  

data_answer = pd.read_csv("./data/zhihurec/pre/pre_info_answer_small.csv")
data_user = pd.read_csv("./data/zhihurec/pre/pre_info_user_small.csv")
data_impression = pd.read_csv("./data/zhihurec/pre/pre_inter_impression_small.csv")
data_token = pd.read_csv("./data/zhihurec/ori/info_token.csv", header=None)


column_names_user = ['user_ID', 'register_timestamp', 'gender', 'login_frequency', 'followers', 'topics_followed', 
                'questions_followed', 'answers', 'questions', 'comments', 'thanks_received', 'comments_received',
                'likes_received', 'dislikes_received', 'register_type', 'register_platform', 'android_or_not',
                'iphone_or_not', 'ipad_or_not', 'pc_or_not', 'mobile_web_or_not', 'device_model', 'device_brand', 'platform', 'province',
                'city'
                # 'topic_ID'
                ]  

column_names_answer = [
    'answer_ID', 
    'question_ID', 
    'anonymous_or_not', 
    'author_ID_(null_for_anonymous)', 
    'labeled_high-value_answer_or_not', 
    'recommended_by_the_editor_or_not', 
    'create_timestamp', 
    'contain_pictures_or_not', 
    'contain_videos_or_not', 
    '#thanks', 
    '#likes', 
    '#comments', 
    '#collections', 
    '#dislikes', 
    '#reports', 
    '#helpless', 
    'token_IDs'
    # 'topic_ID'
]


column_names_impression = ['user_ID', 'answer_ID', 'impression_timestamp', 'click_timestamp'] 

def category_code(columns_to_convert, data):
    data_df = data.copy()
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
    return data_df, column_encoding, feature_lists, data_list

topic_user = data_user[['user_ID', 'topic_ID']].copy()
topic_answer = data_answer[['answer_ID', 'topic_ID']].copy()
feature_lists_topic = {}
column_encoding = {}
unique_topic = get_unique_elements(data_user["topic_ID"], data_answer['topic_ID'])
encoding_topic = {value: i for i, value in enumerate(unique_topic)}
feature_lists_topic["topic_ID"] = [value for value in unique_topic]
column_encoding["topic_ID"] = encoding_topic

topic_user,_,_,_ = category_code(['user_ID'], topic_user)
topic_answer,_,_,_ = category_code(['answer_ID'], topic_answer)
topic_user['topic_ID'] = topic_user['topic_ID'].map(encoding_topic)
topic_answer['topic_ID'] = topic_answer['topic_ID'].map(encoding_topic)

data_list_user_topic = {'user_ID': topic_user['user_ID'].tolist(),
    'topic_ID': topic_user['topic_ID'].tolist()}

data_list_answer_topic = {'answer_ID': topic_answer['answer_ID'].tolist(),
    'topic_ID': topic_answer['topic_ID'].tolist()}

data_answer_tran, column_encoding_answer, feature_lists_answer, data_list_answer = category_code(column_names_answer, data_answer)
data_user_tran, column_encoding_user, feature_lists_user, data_list_user = category_code(column_names_user, data_user)

data_click_impression = click_answer(data_impression)
data_click_impression['user_ID'] = data_click_impression['user_ID'].map(column_encoding_user['user_ID'])
data_click_impression['answer_ID'] = data_click_impression['answer_ID'].map(column_encoding_answer['answer_ID'])
data_list_click_impression = {'user_ID': data_click_impression['user_ID'].tolist(),
                              'answer_ID': data_click_impression['answer_ID'].tolist()}
_, column_encoding_time_impression, feature_lists_time_impression, data_list_time_impression = category_code(['impression_timestamp', 'click_timestamp'], data_click_impression)
data_list_click_impression.update(data_list_time_impression)

data_dict={}

star_graph_edge(data_dict, 'answer_ID', column_names_answer, data_list_answer)
star_graph_edge(data_dict, 'user_ID', column_names_user, data_list_user)
star_graph_edge(data_dict, 'answer_ID', ['answer_ID', 'topic_ID'], data_list_answer_topic)
star_graph_edge(data_dict, 'user_ID', ['user_ID', 'topic_ID'], data_list_user_topic)
complete_graph_edge(data_dict, column_names_impression, data_list_click_impression)

hetero_graph = dgl.heterograph(data_dict).to(device)
print(hetero_graph.number_of_nodes('user_ID'))

def add_feature(hetero_graph, columns_to_convert, feature_lists):
    for column in columns_to_convert:
        
        if column in ['token_IDs'] :
            feature_index = feature_lists[column]
            series = torch.zeros(hetero_graph.number_of_nodes(column), 64)
            for i, element in enumerate(feature_index):
                if element != -1:
                    series[i] = torch.tensor([float(x) for x in data_token.iloc[element, 1].split()])
            hetero_graph.nodes[column].data['feat'] =series.to(device)
        
        elif column in ['topic_ID', 'user_ID', 'answer_ID','author_ID_(null_for_anonymous)', 'question_ID', '#collections', '#comments', '#dislikes', '#helpless', '#likes', '#reports', '#thanks', 'answers', 'comments', 
                         'comments received', 'followers', 'questions_followed', 'thanks_received', 'topics_followed', 'likes_received', 'dislikes_received']:
            feature_list = feature_lists[column]
            series = pd.Series(feature_list)
            series = pd.DataFrame(series)
            hetero_graph.nodes[column].data['feat'] = torch.tensor(series.values, dtype=torch.float32).to(device)
        
        else:
            feature_list = feature_lists[column]
            series = pd.Series(feature_list)
            encoded_features = pd.get_dummies(series, dummy_na=True) 
            hetero_graph.nodes[column].data['feat'] = torch.tensor(encoded_features.values, dtype=torch.float32).to(device)

add_feature(hetero_graph, column_names_user, feature_lists_user)
add_feature(hetero_graph, column_names_answer, feature_lists_answer)

add_feature(hetero_graph, ["topic_ID"], feature_lists_topic)
print(hetero_graph.nodes['topic_ID'].data['feat'])

add_feature(hetero_graph, ['impression_timestamp', 'click_timestamp'],  feature_lists_time_impression)

impression_user_counts = data_click_impression['user_ID'].value_counts()
impression_answer_counts = data_click_impression['answer_ID'].value_counts()

print(impression_user_counts)
print(impression_answer_counts)

for element, count in impression_user_counts.items():
    if 0 <= count <= 19:
        data_user.loc[data_user['user_ID'] == element, 'label_user'] = 0
    elif 20 <= count <= 49:
        data_user.loc[data_user['user_ID'] == element, 'label_user'] = 1
    elif 50 <= count <= 99:
        data_user.loc[data_user['user_ID'] == element, 'label_user'] = 2
    elif 100 <= count:
        data_user.loc[data_user['user_ID'] == element, 'label_user'] = 3

for element, count in impression_answer_counts.items():
    if 0 <= count <= 9:
        data_answer.loc[data_answer['answer_ID'] == element, 'label_answer'] = 0
    elif 10 <= count <= 99:
        data_answer.loc[data_answer['answer_ID'] == element, 'label_answer'] = 1
    elif 100 <= count <= 199:
        data_answer.loc[data_answer['answer_ID'] == element, 'label_answer'] = 2
    elif 200 <= count:
        data_answer.loc[data_answer['answer_ID'] == element, 'label_answer'] = 3


data_user['label_user'].fillna(4, inplace=True)
data_answer['label_answer'].fillna(4, inplace=True)
print(data_user['label_user'])
print(data_answer['label_answer'])


hetero_graph.nodes['user_ID'].data['label_user'] =torch.tensor(data_user['label_user'].tolist(), dtype=torch.float32).to(device)
hetero_graph.nodes['answer_ID'].data['label_answer'] =torch.tensor(data_answer['label_answer'].tolist(), dtype=torch.float32).to(device)


print(hetero_graph.nodes['user_ID'])
print(hetero_graph.nodes['answer_ID'])


js_list = [column_encoding_answer, column_encoding_user, column_encoding, column_encoding_time_impression, 
           feature_lists_answer, feature_lists_user, feature_lists_topic,feature_lists_time_impression,
           data_list_answer, data_list_user, data_list_answer_topic, data_list_user_topic, data_list_click_impression, data_list_time_impression]

def convert_to_builtin_type(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

for js in js_list:
    var_name = [var for var in globals() if globals()[var] is js][0]
    file_name = './data/zhihurec/datadictionary/' + var_name + '.json'
    
    with open(file_name, 'w') as file:
        json.dump(str(js), file)

node_types = hetero_graph.ntypes

for ntype in node_types:
    num_nodes = hetero_graph.number_of_nodes(ntype)
    print(f"Number of nodes in '{ntype}': {num_nodes}")

for node_type in node_types:
    nodes = hetero_graph.nodes(node_type)
    print(f"Node type: {node_type}")

    for node in nodes[:2]:
        feat = hetero_graph.nodes[node_type].data['feat'][node]
        print(f"Node: {node}, Feat: {feat}")

    print()

dgl.save_graphs("2_bhpre_Data_hetero_graph.dgl", [hetero_graph])

print('------------------finish-------------------')
