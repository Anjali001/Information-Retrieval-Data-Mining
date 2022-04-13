#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[23]:


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


# In[7]:


candidate_passages_all = pd.read_csv('candidate-passages-top1000.tsv',sep='\t',names=['qid','pid','query','passage'])
candidate_passages_unique = candidate_passages_all.drop_duplicates(subset=['pid'], inplace=False)
N = len(candidate_passages_unique)


# In[45]:


candidate_passages_all.head()


# In[46]:


candidate_passaged_pid_qid = candidate_passages_all[['qid','pid']]


# In[47]:


candidate_passaged_pid_qid.head()


# In[8]:


inverted_index = {}

for index, data in candidate_passages_unique.iterrows():
    pid = data['pid']
    tokens = preprocess_single_passage(data['passage'],stop_words=True)
    freq_tokens = nltk.FreqDist(tokens)
    words_passage = len(tokens)
    for token, freq in freq_tokens.items():
        inverted_index.setdefault(token, [])
        inverted_index[token].append((pid, freq, words_passage))


# In[76]:


dict(list(inverted_index.items())[0:2])

