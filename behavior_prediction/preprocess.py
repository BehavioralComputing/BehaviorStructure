import pandas as pd


def replaceColumnNames(data, column_names):
    data.columns = column_names
    return data

def timeBox(data, cloumnName, box):
    if box == 'day': 
        time = 'D'
        data[cloumnName] = pd.to_datetime(data[cloumnName], unit='s').dt.floor(time).values.astype('datetime64[s]').view('int64')
    elif box == 'hour': 
        time = 'H'
        data[cloumnName] = pd.to_datetime(data[cloumnName], unit='s').dt.floor(time).values.astype('datetime64[s]').view('int64')
    elif box == 'min': 
        time = 'T'
        data[cloumnName] = pd.to_datetime(data[cloumnName], unit='s').dt.floor(time).values.astype('datetime64[s]').view('int64')
    elif box == 'month': 
        data[cloumnName] = pd.to_datetime(data[cloumnName], unit='s').dt.to_period('M').dt.to_timestamp().astype('datetime64[s]').view('int64')
    elif box == 'week': 
        data[cloumnName] = pd.to_datetime(data[cloumnName], unit='s').dt.to_period('W').dt.to_timestamp().astype('datetime64[s]').view('int64')
    
    return data

def extract_first_number(data, column):
    data[column] = data[column].str.split().str[0]
    return data
path_user = './data/zhihurec/ori/info_user_small.csv'
path_answer = './data/zhihurec/ori/info_answer_small.csv'
path_impression = './data/zhihurec/ori/inter_impression_small.csv'

data_user = pd.read_csv(path_user) 
data_answer =  pd.read_csv(path_answer) 
data_impression = pd.read_csv(path_impression) 

column_names_user = ['user_ID', 'register_timestamp', 'gender', 'login_frequency', 'followers', 'topics_followed', 
                'questions_followed', 'answers', 'questions', 'comments', 'thanks_received', 'comments received',
                'likes_received', 'dislikes_received', 'register_type', 'register_platform', 'android_or_not',
                'iphone_or_not', 'ipad_or_not', 'pc_or_not', 'mobile_web_or_not', 'device_model', 'device_brand', 'platform', 'province',
                'city', 'topic_ID']  
data_user = replaceColumnNames(data_user, column_names_user)
data_user = timeBox(data_user, 'register_timestamp', 'month')
data_user = extract_first_number(data_user, 'topic_ID')
data_user = data_user.fillna(-1)

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
    'token_IDs', 
    'topic_ID'
]
data_answer = replaceColumnNames(data_answer, column_names_answer)
data_answer = timeBox(data_answer, 'create_timestamp', 'week')
data_answer = extract_first_number(data_answer, 'token_IDs')
data_answer = extract_first_number(data_answer, 'topic_ID')
data_answer = data_answer.fillna(-1)

column_names_impression = ['user_ID', 'answer_ID', 'impression_timestamp', 'click_timestamp']

data_impression = replaceColumnNames(data_impression, column_names_impression)
data_impression = timeBox(data_impression, 'impression_timestamp', 'hour')
data_impression = timeBox(data_impression, 'click_timestamp', 'hour')
data_impression = data_impression.fillna(-1)

data_user.to_csv('./data/zhihurec/pre/pre_info_user_small.csv', index=False)
data_answer.to_csv('./data/zhihurec/pre/pre_info_answer_small.csv', index=False)
data_impression.to_csv('./data/zhihurec/pre/pre_inter_impression_small.csv', index=False)

print('-----------------finish---------------')