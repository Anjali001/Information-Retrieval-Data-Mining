# -*- coding: utf-8 -*-
"""IRDM_cw2_Part4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o6bQa_pT77HuQpglj1OIPmzBA71UDCT6
"""

import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
pd.options.mode.chained_assignment = None
from tqdm import tqdm
from imblearn import under_sampling

from google.colab import drive
drive.mount('/content/drive')

validation_data = pd.read_csv('/content/drive/MyDrive/IRDM Coursework 2/validation_data_cosine.csv',index_col=0)

final_train = pd.read_csv('/content/drive/MyDrive/IRDM Coursework 2/train_data_cosine.csv',index_col=0)

final_train['DocLen'] = 0
final_train['queryLen'] = 0
for i, row in final_train.iterrows():
  final_train['DocLen'][i] = len(final_train['passage'][i])
  final_train['queryLen'][i] = len(final_train['queries'][i])

validation_data['DocLen'] = 0
validation_data['queryLen'] = 0
for i, row in validation_data.iterrows():
  validation_data['DocLen'][i] = len(validation_data['passage'][i])
  validation_data['queryLen'][i] = len(validation_data['queries'][i])

t = under_sampling.RandomUnderSampler(random_state=42)

x_train = final_train[['cosine_similarity', 'DocLen', 'queryLen']].values
y_train = final_train[['relevancy']].values
x_train,y_train = t.fit_resample(x_train,y_train)

x_test = validation_data[['cosine_similarity', 'DocLen', 'queryLen']].values
y_test = validation_data[['relevancy']].values

# from sklearn.model_selection import train_test_split
# x_train,x_val,y_train,y_val=train_test_split(x,Y,test_size=0.10,shuffle=True,stratify = Y)

from sklearn.preprocessing import StandardScaler
##Normalization is done so that the difference between highest and lowest data point is not too large
import numpy as np
scaler=StandardScaler()
##To find mean and std dev
scaler.fit(x_train)
## Please Note - We are using mean and std dev of training data for testing data too. It is done to ensure that no information of test data is leaked to the model.
##Converting data into form where mean of data is 0 and std dev is 1 
X_train=scaler.transform(x_train)
X_test=scaler.transform(x_test)

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim=X_train.shape[1], activation='relu'))
##Dropout is used to avoid overfitting
keras.layers.Dropout(0.2)
model.add(Dense(20, activation='relu'))
keras.layers.Dropout(0.2)
model.add(Dense(20, activation='relu'))
keras.layers.Dropout(0.2)
##For classification problems, we usually use softmax as activation function in final layer
model.add(Dense(1, activation='sigmoid')) 
##Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision()])

model.summary()



model.fit(x_train, y_train, epochs=8, batch_size=80, validation_split=0.1)

testing=model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1]+'uracy of Model on testing data', testing[1]*100))

from sklearn.metrics import classification_report, confusion_matrix  
predictions = model.predict(X_test)

validation_data['score'] = predictions
validation_data.head()

validation_data['NN_rank'] = validation_data.groupby('qid')['score'].rank(method='first',ascending=False).astype('int')

validation_data.head()

trial_data = validation_data[['qid','pid','NN_rank','score']]

trial_data = trial_data.reset_index(drop=True)

NN_dict = {}
qid_list = trial_data['qid'].unique()
for qid in qid_list:
    top_ones = trial_data[trial_data['qid'] == qid]
    top_ones = top_ones.reset_index(drop=True)
    top_ones = top_ones.sort_values(by=['NN_rank'])
    NN_dict[qid] = top_ones[:100]

f = open("NN.txt", "w")
for lr_df in NN_dict.values():
    for i, data in lr_df.iterrows():
        qid = str(data['qid'].astype(int))
        pid = str(data['pid'].astype(int))
        score = str(data['score'])
        rank = str(data['NN_rank'].astype(int))
        f.write(qid + "," + "A2" + "," + pid + "," + rank + "," + score + "," + "NN" + "\n")
f.close()

def average_precision_calc(df,retrieved,score,rank):
    average_precision = 0
    qid_list = np.unique(np.asarray(df['qid']))
    ranked_passages = df[df[rank] <= retrieved]

    relevant_passage = ranked_passages[ranked_passages['relevancy'] != 0]
    relevant_passage['rank'] = relevant_passage.groupby('qid')[score].rank(method = 'first',ascending=False)

    for qid in qid_list:
        temp = relevant_passage[relevant_passage['qid'] == qid]
        temp['rank'] = temp['rank']/temp[rank]
        if len(temp) == 0:
            average_precision += 0
        else:
            average_precision += sum(temp['rank'])/len(temp)

    average_precision = average_precision/len(qid_list)
    return average_precision

average_precision_calc(validation_data,100,'score','NN_rank')

def NDCG_calc(df,retrieved, rank):

    all_DCG = 0
    relevant_passage = df[df['relevancy'] != 0]
    relevant_passage_retrived = relevant_passage[relevant_passage[rank] <= retrieved]

    qid_list = np.unique(np.asarray(df['qid']))

    for qid in qid_list:
        temp = relevant_passage[relevant_passage['qid'] == qid]
        DCG = sum(1/np.log2(np.asarray(temp[rank])+1))
        optDCG = sum(1/np.log2(np.arange(1,len(temp)+1)+1))
        all_DCG += DCG/optDCG
    all_DCG = all_DCG/len(qid_list)

    return all_DCG

NDCG_calc(validation_data,100,'NN_rank')



# import tensorflow as tf
# from tensorflow.keras import layers, activations, losses, Model, Input
# from tensorflow.nn import leaky_relu
# import numpy as np
# from itertools import combinations
# from tensorflow.keras.utils import plot_model, Progbar
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from torch import nn

# qids = final_train.qid.values
# doc_features = final_train[['cosine_similarity', 'DocLen', 'queryLen']].values
# doc_scores = final_train.relevancy.values

# # put data into pairs
# xi = []
# xj = []
# pij = []
# pair_id = []
# pair_query_id = []
# for q in np.unique(qids):
#     query_idx = np.where(qids == q)[0]
#     for pair_idx in combinations(query_idx, 2):
#         pair_query_id.append(q)
        
#         pair_id.append(pair_idx)
#         i = pair_idx[0]
#         j = pair_idx[1]
#         xi.append(doc_features[i])
#         xj.append(doc_features[j])
        
#         if doc_scores[i] == doc_scores[j]:
#             _pij = 0.5
#         elif doc_scores[i] > doc_scores[j]:
#             _pij = 1
#         else: 
#             _pij = 0
#         pij.append(_pij)
        
# xi = np.array(xi)
# xj = np.array(xj)
# pij = np.array(pij)
# pair_query_id = np.array(pair_query_id)

# xi_train, xi_test, xj_train, xj_test, pij_train, pij_test, pair_id_train, pair_id_test = train_test_split(
#     xi, xj, pij, pair_id, test_size=0.2, stratify=pair_query_id)

# qidsV = validation_data.qid.values
# doc_featuresV = validation_data[['cosine_similarity', 'DocLen', 'queryLen']]
# doc_scoresV = validation_data.relevancy.values

# doc_featuresV.head()

# xiV = []
# xjV = []
# # pijV = []
# # pair_idV = []
# # pair_query_idV = []
# uniqeu_qids = np.unique(qidsV)
# for i,q in enumerate(tqdm(uniqeu_qids[:1])):
#     query_idx = np.where(qidsV == q)[0]
#     for pair_idx in combinations(query_idx, 2):
#         # pair_query_idV.append(q)
        
#         # pair_idV.append(pair_idx)
#         i = pair_idx[0]
#         j = pair_idx[1]
#         xiV.append(doc_featuresV[i])
#         xjV.append(doc_featuresV[j])
        
#         # if doc_scoresV[i] == doc_scoresV[j]:
#         #     _pijV = 0.5
#         # elif doc_scoresV[i] > doc_scoresV[j]:
#         #     _pijV = 1
#         # else: 
#         #     _pijV = 0
#         # pijV.append(_pijV)
        
# xiV = np.array(xiV)
# xjV = np.array(xjV)
# # pair_query_idV = np.array(pair_query_idV)

# class RankNet(Model):
#     def __init__(self):
#         super().__init__()
#         self.dense = [layers.Dense(16, activation=leaky_relu), layers.Dense(8, activation=leaky_relu)]
#         self.o = layers.Dense(1, activation='linear')
#         self.oi_minus_oj = layers.Subtract()
    
#     def call(self, inputs):
#         xi, xj = inputs
#         densei = self.dense[0](xi)
#         densej = self.dense[0](xj)
#         for dense in self.dense[1:]:
#             densei = dense(densei)
#             densej = dense(densej)
#         oi = self.o(densei)
#         oj= self.o(densej)
#         oij = self.oi_minus_oj([oi, oj])
#         output = layers.Activation('sigmoid')(oij)
#         return output

# ranknet = RankNet()
# ranknet.compile(optimizer='adam', loss='binary_crossentropy')
# history = ranknet.fit([xi_train, xj_train], pij_train, epochs=10, batch_size=1000, validation_data=([xi_test, xj_test], pij_test))

# ranknet.summary()

# ranknet.call([xiV, xjV])

# print(y_pred)



