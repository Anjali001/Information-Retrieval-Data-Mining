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


inverted_index = {}

for index, data in candidate_passages_unique.iterrows():
    pid = data['pid']
    tokens = preprocess_single_passage(data['passage'],stop_words=True)
    freq_tokens = nltk.FreqDist(tokens)
    words_passage = len(tokens)
    for token, freq in freq_tokens.items():
        inverted_index.setdefault(token, [])
        inverted_index[token].append((pid, freq, words_passage))


# In[8]:


dict(list(inverted_index.items())[0:2])


# In[9]:


vocab = list(inverted_index.keys())
total_length_vocab = len(vocab)


# In[10]:


############### for every word, I have a representation


# In[11]:


tf_idf = {} # instead of a 1.5m X 1.5m matrix for passages throws memory error, making a dictionary -> will access it via pid,token
for token, pid_freq_words in inverted_index.items():
    nt = len(pid_freq_words) # Number of docs word occured in
    for pid, freq_in_pid, pid_words in pid_freq_words:
        term_freq = freq_in_pid/pid_words
        idf = np.log10(N/nt)
        tf_idf[pid, token] = term_freq*idf # Make a seq like for one word, all pid and calculate tfidf.... 


# In[12]:


dict(list(tf_idf.items())[0:100])


# In[13]:


def query_vec(tokens):
    query_length = len(tokens)
    query_vect = np.zeros((total_length_vocab))
    query_freq_dist = nltk.FreqDist(tokens)
    for token in np.unique(tokens):
        tf = query_freq_dist[token]/query_length
        try:
            df = len(inverted_index[token])
            idf = np.log10(N/df)
            index_of_token = vocab.index(token)
            query_vect[index_of_token] = tf*idf
        except:
            pass
    return query_vect
        
#         if token not  in inverted_index.keys():
#             pass
#         else:
#             df = len(inverted_index[token])
#             idf = np.log10(N/df)
#             index_of_token = vocab.index(token)
#             query_vect[index_of_token] = tf*idf
#     return query_vect
        
#         try:
#             df = len(inverted_index[token])
#         except:
#             df=1 #down
#         idf = np.log10(N/df)
#         index_of_token = vocab.index(token)
#         query_vect[index_of_token] = tf*idf
#     return query_vect


# In[14]:


def passage_vec(tokens,pid):
    passage_vect = np.zeros((total_length_vocab))
    for token in np.unique(tokens):
        try:
            index_of_token = vocab.index(token)
            passage_vect[index_of_token] = tf_idf[(pid,token)]
        except:
            pass
    return passage_vect
            
#         if token not in vocab:
#             pass
#         else:
#             index_of_token = vocab.index(token)
#             passage_vect[index_of_token] = tf_idf[(pid,token)]
    return passage_vect


# In[15]:


def cosine_sim(query, passage, pid):
    query_tokens = preprocess_single_passage(query)
    passage_tokens = preprocess_single_passage(passage)
    query_vector = query_vec(query_tokens)
    passage_vector = passage_vec(passage_tokens, pid)
    
    cosine = np.dot(query_vector, passage_vector)/ np.linalg.norm(query_vector)*np.linalg.norm(passage_vector)
    
    return cosine


# In[16]:


test_queries = pd.read_csv("test-queries.tsv", sep='\t', names=["qid","query"])


# In[17]:


test_queries.head()


# In[18]:


cosine_dictionary = {}
for index, query in test_queries.iterrows():
    qid = query['qid']
    cosine_dictionary[qid] = []
    candidated_passages = candidate_passages_all[candidate_passages_all['qid'] == qid]
    for ind, query in candidated_passages.iterrows():
        pid = query['pid']
        passage = query['passage']
        query = query['query']
        cosine_dictionary[qid].append(cosine_sim(query, passage, pid))


# In[19]:


dict(list(cosine_dictionary.items())[0:1])


# In[20]:


def sim_rank(cosine_sim_results):
    result = np.array(cosine_sim_results).argsort()[-100:][::-1]
    return result


# In[22]:


results_cosine = {}
for idx, tuplee in test_queries.iterrows():
    qid = tuplee['qid']
    results_cosine[qid] = sim_rank(cosine_dictionary[qid]) # for every qid getting top 100 


# In[23]:


dict(list(results_cosine.items())[0:1])


# In[25]:


tf_idf_dict = {}
for qid, indices in results_cosine.items():
    passaged_100 = candidate_passaged_pid_qid[candidate_passaged_pid_qid['qid'] == qid] # Get all passages for one qid
    passaged_100 = passaged_100.reset_index(drop=True)
    tf_idf_sort_indices = passaged_100.loc[indices]
    tf_idf_sort_indices = tf_idf_sort_indices.reset_index(drop=True)
    tf_idf_sort_indices['score'] = np.array(cosine_dictionary[qid])[indices]
    tf_idf_dict[qid] = tf_idf_sort_indices


# In[26]:


for x in tf_idf_dict.values():
    print(type(x))
    break


# In[27]:


f = open("tfidf.csv", "w")
for pd_df in tf_idf_dict.values():
    for rank, data in pd_df.iterrows():
        qid = str(data['qid'])
        pid = str(data['pid'])
        cosine = str(data['score'])
        f.write(qid + "," + pid + "," + cosine + "\n")
f.close()


# In[28]:


word_occur_corpus = 0
for idx, data in candidate_passages_unique.iterrows():
    word_occur_corpus += len(preprocess_single_passage(data['passage']))
avg_passage_len = word_occur_corpus/N


# In[29]:


print(avg_passage_len)


# In[30]:


k1 = 1.2
k2 = 100
b = 0.75
R = 0
r = 0


# In[31]:


def BM25_model(query, passage):
    p_tokens = preprocess_single_passage(passage)
    q_tokens = preprocess_single_passage(query)
    
    q_length = len(q_tokens)
    
    passage_freq_dist = nltk.FreqDist(p_tokens)
    query_freq_dist = nltk.FreqDist(q_tokens)
    
    
    doclen = len(p_tokens)
    K = k1*((1-b) + b *(float(doclen)/float(avg_passage_len)))
    
    score = 0
    for token in q_tokens:
        try:
            n = len(inverted_index[token])
        except:
            n = 0
        f = passage_freq_dist[token]
        qf = query_freq_dist[token]
        one = np.log(((r + 0.5)/(R - r + 0.5))/((n-r+0.5)/(N-n-R+r+0.5)))
        two = ((k1 + 1) * f)/(K+f)
        three = ((k2+1) * qf)/(k2+qf)
        score += one * two * three
    return score


# In[33]:


bm25_dict = {}
for idx, data in test_queries.iterrows():
    qid = data['qid']
    bm25_dict[qid] = []
    candidate_passaged_ = candidate_passages_all[candidate_passages_all['qid'] == qid]
    for idx_P, data_P in candidate_passaged_.iterrows():
        passage = data_P['passage']
        query = data_P['query']
        bm25_dict[qid].append(BM25_model(query, passage))


# In[35]:


dict(list(bm25_dict.items())[0:1])


# In[36]:


results_bm25 = {}
for index, data in test_queries.iterrows():
    qid = data['qid']
    results_bm25[qid] = sim_rank(bm25_dict[qid])


# In[38]:


bm_25_final_dict = {}
for qid, args in results_bm25.items():
    passages_top1000 = candidate_passages_all[candidate_passages_all['qid'] == qid]
    passages_top1000 = passages_top1000.reset_index(drop=True)
    bm25_top100 = passages_top1000.loc[args]
    bm25_top100 = bm25_top100.reset_index(drop=True)
    bm25_top100['score'] = np.array(bm25_dict[qid])[args]
    bm_25_final_dict[qid] = bm25_top100


# In[39]:


f = open("bm25.csv", "w")
for bm_df in bm_25_final_dict.values():
    for rank, data in bm_df.iterrows():
        qid = str(data['qid'])
        pid = str(data['pid'])
        bm = str(data['score'])
        f.write(qid + "," + pid + "," + bm + "\n")
f.close()

