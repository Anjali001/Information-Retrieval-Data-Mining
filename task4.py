#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from tqdm import tqdm


# In[2]:


def preprocess_single_passage(passage,stop_words=True):
    tokenizer = RegexpTokenizer(r'\w+')
    passage = passage.lower()
    tok_pass = tokenizer.tokenize(passage)
    tok_pass = [tok for tok in tok_pass if tok.isalpha()]
    if stop_words == True:
        stop_words = stopwords.words('english')
        tokens = [tok for tok in tok_pass if tok not in stop_words]
    else:
        tokens = tok_pass
    return tokens


# In[3]:


candidate_passages_all = pd.read_csv('candidate-passages-top1000.tsv',sep='\t',names=['qid','pid','query','passage'])
candidate_passages_unique = candidate_passages_all.drop_duplicates(subset=['pid'], inplace=False)
N = len(candidate_passages_unique)


# In[4]:


candidate_passages_all.head()


# In[5]:


candidate_passaged_pid_qid = candidate_passages_all[['qid','pid']]


# In[6]:


candidate_passaged_pid_qid.head()


# In[7]:


test_queries = pd.read_csv("test-queries.tsv", sep='\t', names=["qid","query"])


# In[8]:


test_queries.head()


# In[9]:


def sim_rank(cosine_sim_results):
    result = np.array(cosine_sim_results).argsort()[-100:][::-1]
    return result


# In[10]:


inverted_index = {}

for index, data in candidate_passages_unique.iterrows():
    pid = data['pid']
    tokens = preprocess_single_passage(data['passage'],stop_words=True)
    freq_tokens = nltk.FreqDist(tokens)
    words_passage = len(tokens)
    for token, freq in freq_tokens.items():
        inverted_index.setdefault(token, [])
        inverted_index[token].append((pid, freq, words_passage))


# In[11]:


vocab = list(inverted_index.keys())
V = len(vocab)


# In[12]:


def sim_rank(cosine_sim_results):
    result = np.array(cosine_sim_results).argsort()[-100:][::-1]
    return result


# In[13]:


def modelling_laplace(query, passage):
    q_tokens = preprocess_single_passage(query)
    p_tokens = preprocess_single_passage(passage)
    D = len(p_tokens)
    p_fdist = nltk.FreqDist(p_tokens)
    score = 0
    for token in q_tokens:
        score += np.log((p_fdist[token]+1)/(D+V))     
    return score

def modelling_lidstone(query, passage):
    q_tokens = preprocess_single_passage(query)
    p_tokens = preprocess_single_passage(passage)
    D = len(p_tokens)
    p_fdist = nltk.FreqDist(p_tokens)
    score = 0
    epsilon = 0.1
    for token in q_tokens:
        score += np.log((p_fdist[token]+epsilon)/(D+epsilon*V))
    return score


# In[15]:


laplace_dict = {}
for index, data in test_queries.iterrows():
    qid = data['qid']
    laplace_dict[qid] = []
    passages_top1000 = candidate_passages_all[candidate_passages_all['qid'] == qid]
    for idx_P, data_P in passages_top1000.iterrows():
        passage = data_P['passage']
        query = data_P['query']
        laplace_dict[qid].append(modelling_laplace(query, passage))
        
results_laplace = {}
for index, data in test_queries.iterrows():
    qid = data['qid']
    results_laplace[qid] = sim_rank(laplace_dict[qid])
    
laplace_final_dict = {}
for qid, args in results_laplace.items():
    passages_top1000 = candidate_passages_all[candidate_passages_all['qid'] == qid]
    passages_top1000 = passages_top1000.reset_index(drop=True)
    laplace_top100 = passages_top1000.loc[args]
    laplace_top100 = laplace_top100.reset_index(drop=True)
    laplace_top100['score'] = np.array(laplace_dict[qid])[args]
    laplace_final_dict[qid] = laplace_top100

f = open("laplace.csv", "w")
for bm_df in laplace_final_dict.values():
    for rank, data in bm_df.iterrows():
        qid = str(data['qid'])
        pid = str(data['pid'])
        bm = str(data['score'])
        f.write(qid + "," + pid + "," + bm + "\n")
f.close()


# In[17]:


lidstone_dict = {}
for index, data in test_queries.iterrows():
    qid = data['qid']
    lidstone_dict[qid] = []
    passages_top1000 = candidate_passages_all[candidate_passages_all['qid'] == qid]
    for index_P, data_P in passages_top1000.iterrows():
        passage = data_P['passage']
        query = data_P['query']
        lidstone_dict[qid].append(modelling_lidstone(query, passage))
        
results_lidstone = {}
for index, data in test_queries.iterrows():
    qid = data['qid']
    results_lidstone[qid] = sim_rank(lidstone_dict[qid])
    
lidstone_final_dict = {}
for qid, args in results_lidstone.items():
    passages_top1000 = candidate_passages_all[candidate_passages_all['qid'] == qid]
    passages_top1000 = passages_top1000.reset_index(drop=True)
    lidstone_top100 = passages_top1000.loc[args]
    lidstone_top100 = lidstone_top100.reset_index(drop=True)
    lidstone_top100['score'] = np.array(lidstone_dict[qid])[args]
    lidstone_final_dict[qid] = lidstone_top100
    
f = open("lidstone.csv", "w")
for bm_df in lidstone_final_dict.values():
    for rank, data in bm_df.iterrows():
        qid = str(data['qid'])
        pid = str(data['pid'])
        bm = str(data['score'])
        f.write(qid + "," + pid + "," + bm + "\n")
f.close()


# In[20]:


def freqq(token):
    freq = 0
    try:
        for tup in inverted_index[token]:
            freq += tup[1]
    except:
        pass
    return freq

def modelling_dirichlet(query, passage):
    q_tokens = preprocess_single_passage(query)
    p_tokens = preprocess_single_passage(passage)
    p_fdist = nltk.FreqDist(p_tokens)
    N = len(p_tokens)
    mu = 50
    score = 0
    for token in q_tokens:
        freq = freqq(token)
        lambdaa = N / (N + mu)
        one_lambda = mu / (N + mu)
        first_term = lambdaa * (p_fdist[token] / N)
        second_term = one_lambda * (freq / V)
        if (first_term + second_term == 0):
            continue
        score += np.log(first_term + second_term)
    return score

dirichlet_dict = {}
for index, data in test_queries.iterrows():
    qid = data['qid']
    dirichlet_dict[qid] = []
    passages_top1000 = candidate_passages_all[candidate_passages_all['qid'] == qid]
    for index_P, data_P in passages_top1000.iterrows():
        passage = data_P['passage']
        query = data_P['query']
        dirichlet_dict[qid].append(modelling_dirichlet(query, passage))
        
results_dirichlet = {}
for index, data in test_queries.iterrows():
    qid = data['qid']
    results_dirichlet[qid] = sim_rank(dirichlet_dict[qid])
    
dirichlet_final_dict = {}
for qid, args in results_dirichlet.items():
    passages_top1000 = candidate_passages_all[candidate_passages_all['qid'] == qid]
    passages_top1000 = passages_top1000.reset_index(drop=True)
    dirichlet_top100 = passages_top1000.loc[args]
    dirichlet_top100 = dirichlet_top100.reset_index(drop=True)
    dirichlet_top100['score'] = np.array(dirichlet_dict[qid])[args]
    dirichlet_final_dict[qid] = dirichlet_top100
    
f = open("dirichlet.csv", "w")
for bm_df in dirichlet_final_dict.values():
    for rank, data in bm_df.iterrows():
        qid = str(data['qid'])
        pid = str(data['pid'])
        bm = str(data['score'])
        f.write(qid + "," + pid + "," + bm + "\n")
f.close()

