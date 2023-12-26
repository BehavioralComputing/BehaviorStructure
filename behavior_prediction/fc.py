import pandas as pd
import os
import json
from tqdm import tqdm

filepath='./data/zhihurec/pre/pre_inter_impression_small.csv'
print(os.path.exists(filepath))
print(os.getcwd())
data=pd.read_csv('./data/zhihurec/ori/inter_impression_small.csv')

dictcolm={
    data.columns[0]:'user_id:token',
    data.columns[1]: 'item_id:token',
     data.columns[2]: 'imp_timestamp:float',
    data.columns[3]: 'timestamp:float',
}

data =data.rename(columns=dictcolm)
data['rating:float'] = data['timestamp:float'].apply(lambda x: 0 if x == 0 else 2)
data.to_csv('./dataset/zhihu_vec/zhihu_vec.inter',sep='	',index=False)

file_path = './data/2_node_embeddings_liner.json'  

with open(file_path, 'r') as file:
    node_embeddings = json.load(file)
    
column1 = node_embeddings['user_ID']  
column2 = node_embeddings['answer_ID']
data_user =pd.read_csv('./data/zhihurec/pre/pre_info_user_small.csv', usecols=[0])
userlist= data_user['user_ID']

data_user['vector_user'] = column1
data_user['vector_user'] = data_user['vector_user'].apply(lambda x: " ".join(str(val) for val in x))

dictcolm = {
    data_user.columns[0]: 'user_id:token',
    data_user.columns[1]: 'vector_user:float_seq'
}

data_user = data_user.rename(columns=dictcolm)
data_user.to_csv('./dataset/zhihu_vec/zhihu_vec.user',sep='	',index=False)
data_item =pd.read_csv('./data/zhihurec/pre/pre_info_answer_small.csv', usecols=[0])
itemlist= data_item['answer_ID']
data_item['vector_answer'] = column2
data_item['vector_answer'] = data_item['vector_answer'].apply(lambda x: " ".join(str(val) for val in x))

dictcolm = {
    data_item.columns[0]: 'item_id:token',
    data_item.columns[1]: 'vector_answer:float_seq'
}

data_item=data_item.rename(columns=dictcolm)
data_item.to_csv('./dataset/zhihu_vec/zhihu_vec.item',sep='	',index=False)

print()