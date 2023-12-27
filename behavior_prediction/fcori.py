
# -*- coding: utf-8 -*-
import pandas as pd
import os
import json
from tqdm import tqdm
import math


data=pd.read_csv('./data/zhihurec/ori/inter_impression_small.csv')

dictcolm={
    data.columns[0]:'user_id:token',
    data.columns[1]: 'answer_id:token',
     data.columns[2]: 'imp_timestamp:float',
    data.columns[3]: 'timestamp:float',
}

data =data.rename(columns=dictcolm)
data['rating:float'] = data['timestamp:float'].apply(lambda x: 0 if x == 0 else 2)
data.to_csv('./dataset/zhihu_ori/zhihu_ori.inter',sep='	',index=False)

data_user =pd.read_csv('./data/zhihurec/ori/info_user_small.csv')

dictcolm = {
    data_user.columns[0]: 'user_id:token',
    data_user.columns[1]: 'register_timestamp:float',
    data_user.columns[2]: 'gender:token',
    data_user.columns[3]: 'login_frequency:float',
    data_user.columns[4]: 'followers:float',
    data_user.columns[5]: 'topics_followed:float',
    data_user.columns[6]: 'questions_followed:float',
    data_user.columns[7]: 'answers:float',
    data_user.columns[8]: 'questions:float',
    data_user.columns[9]: 'comments:float',
    data_user.columns[10]: 'thanks_received:float',
    data_user.columns[11]: 'comments_received:float',
    data_user.columns[12]: 'likes_received:float',
    data_user.columns[13]: 'dislikes_received:float',
    data_user.columns[14]: 'register_type:token',
    data_user.columns[15]: 'register_platform:token',
    data_user.columns[16]: 'android_or_not:token',
    data_user.columns[17]: 'iphone_or_not:token',
    data_user.columns[18]: 'ipad_or_not:token',
    data_user.columns[19]: 'pc_or_not:token',
    data_user.columns[20]: 'mobile_web_or_not:token',
    data_user.columns[21]: 'device_model:token',
    data_user.columns[22]: 'device_brand:token',
    data_user.columns[23]: 'platform:token',
    data_user.columns[24]: 'province:token',
    data_user.columns[25]: 'city:token',
    data_user.columns[26]: 'topic:token_seq',   
}


data_user = data_user.rename(columns=dictcolm)

rows = data_user['topic:token_seq']
for index, row in tqdm(enumerate(data_user['topic:token_seq']), total=len(data_user['topic:token_seq'])):
    if not pd.isna(row):
        numbers = row.split(' ')
        if len(numbers) > 99:
            processed_row = ' '.join(numbers[:100])
            data_user['topic:token_seq'][index] = processed_row
        

        
    





data_user.to_csv('./dataset/zhihu_ori/zhihu_ori.user',sep='	',index=False)
data_item =pd.read_csv('./data/zhihurec/ori/info_answer_small.csv')


dictcolm = {
    data_item.columns[0]: 'answer_id:token',
    data_item.columns[1]: 'question_ID:token',
    data_item.columns[2]: 'anonymous_or_not:token',
    data_item.columns[3]: 'author_ID:token',
    data_item.columns[4]: 'labeled_high:token',
    data_item.columns[5]: 'recommended_by_the_editor_or_not:token',
    data_item.columns[6]: 'create_timestamp:float',
    data_item.columns[7]: 'contain_pictures_or_not:token',
    data_item.columns[8]: 'contain_videos_or_not:token',
    data_item.columns[9]: 'thanks:float',
    data_item.columns[10]: 'likes:float',
    data_item.columns[11]: 'comments:float',
    data_item.columns[12]: 'collections:float',
    data_item.columns[13]: 'dislikes:float',
    data_item.columns[14]: 'reports:float',
    data_item.columns[15]: 'helpless:float',
    data_item.columns[16]: 'token_IDs:token_seq',
    data_item.columns[17]: 'topics:token_seq',
    
}
data_item=data_item.rename(columns=dictcolm)
rows_item_token = data_item['token_IDs:token_seq']
rows_item_topic = data_item['topics:token_seq']
for index, row in tqdm(enumerate(data_item['token_IDs:token_seq']), total=len(data_item['topics:token_seq'])):
    if not pd.isna(row):
        numbers = row.split(' ')
        if len(numbers) > 99:
            processed_row = ' '.join(numbers[:100])
            data_item['token_IDs:token_seq'][index] = processed_row

for index, row in tqdm(enumerate(data_item['topics:token_seq']), total=len(data_item['topics:token_seq'])):
    if not pd.isna(row):
        numbers = row.split(' ')
        if len(numbers) > 99:
            processed_row = ' '.join(numbers[:100])
            data_item['topics:token_seq'][index] = processed_row


data_item.to_csv('./dataset/zhihu_ori/zhihu_ori.item',sep='	',index=False)

print()
