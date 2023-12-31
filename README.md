
# Usage

Download datasets from [Crime Data] (https://catalog.data.gov/dataset/crime-data-from-2020-to-present). And put the files in `./crime_detection/data`.
And run `data_pre_process.py` and `behavioral_space_generation.py` to generate the crime behavioral structure graph.

Download datasets from [ZhihuRec Data] (https://github.com/THUIR/ZhihuRec-Dataset). And put the files in `./behavior_prediction/data/zhihurec/ori`.
Following the article "A Large-Scale Rich Context Query and Recommendation Dataset in Online Knowledge-Sharing (https://arxiv.org/pdf/2106.06467.pdf)," 
data from files `info_user.csv`, `info_answer.csv`, and `inter_impression.csv` were extracted, specifically the first 7,963, 81,214, and 1,000,026 records, respectively. 
These extracted records were then generated into files `info_user_small.csv`, `info_answer_small.csv`, `inter_impression_small.csv`, which were placed in the directory path `./behavior_prediction/data/zhihurec/ori`.
And run `preprocess.py` and `bsg.py` to generate the zhihu behavioral structure graph, run `rgcn.py`, `fc.py` and `fcori.py` to generate `zhihu_vec` and `zhihu_ori` datasets for RecBole.

Download datasets from [Fraudulent Transaction Data] (https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data). And put the files in `./graph_Gen`.
