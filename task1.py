#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


# In[26]:


with open("passage-collection.txt") as whole_text:
  passage = whole_text.readlines()


# In[60]:


def preprocess_data(data,stop_words=False):
  final_sentences=[]
  tokenizer = RegexpTokenizer(r'\w+')
  for sentence in passage:
    tok_sent = tokenizer.tokenize(sentence)
    sentence = [x.lower() for x in tok_sent]
    no_number_sentence = [tok for tok in sentence if tok.isalpha()]
  
    if stop_words == True:
      stop_words = stopwords.words('english')
      no_stopwords_sentence = [tok for tok in no_number_sentence if tok not in stop_words]
      final_sentences.append(no_stopwords_sentence)
    else:
      final_sentences.append(no_number_sentence)
  return final_sentences


# In[67]:


sentenced_token = preprocess_data(passage)


# In[68]:


words = [token for sentence in sentenced_token for token in sentence]


# In[70]:


vocab = list(set(words))


# In[71]:


total_words = len(words)


# In[74]:


def freq(vocab):
  dict1 = {}
  for word in vocab:
    if word in dict1:
      dict1[word] = dict1[word]+1
    else:
      dict1[word]= 1
  return dict1


# In[75]:


frequency_table = freq(words)


# In[76]:


ranked_freq_table = sorted(frequency_table.items(),key = lambda x: x[1], reverse = True)


# In[77]:


summ=0
for i in range(1,total_words+1):
  summ+=(1/i)

c = 1/summ


# In[78]:


final_list = []
rank_list = []
prob_list =[]
for i in range(len(ranked_freq_table)):
  rank_list.append(i+1)
  prob_list.append(float(ranked_freq_table[i][1]/total_words))
  #final list = [Rank,word, frequency, normalised freq ,normfreq*rank]
  final_list.append((i+1,ranked_freq_table[i][0],ranked_freq_table[i][1],
                     float(ranked_freq_table[i][1]/total_words), c/(i+1)))


# In[79]:


p = pd.DataFrame(final_list,columns = ['Rank','word', 'frequency', 'normalisedFreq' ,'normfreq*rank'])


# In[80]:


p.tail()


# In[81]:


p['normfreq*rank'].mean()


# In[ ]:


p['normfreq*rank'].std()


# In[ ]:


p['normalisedFreq'].mean()


# In[ ]:


p['normalisedFreq'].std()


# In[82]:


plt.figure(figsize=(10,5))
plt.plot(p['Rank'],p['normalisedFreq'],label="Frequency per Rank")
plt.plot(p['Rank'],p['normfreq*rank'],label="Frequency per Rank")
plt.legend(['Data','Zipf'])
plt.title("Empirical Distrbution Vs Zipf's Distribution")
plt.xlabel("rank")
plt.ylabel("word frequency")
plt.show()


# In[83]:


plt.figure(figsize=(5,5))
plt.plot(p['Rank'],p['normalisedFreq'],label="Frequency per Rank")
plt.plot(p['Rank'],p['normfreq*rank'],label="Frequency per Rank")
plt.legend(['Data','Zipf'])
plt.title("Log-Log Plot : Empirical Distrbution Vs Zipf's Distribution")
plt.xlabel("rank")
plt.ylabel("word frequency")
plt.xscale('log')
plt.yscale('log')
plt.show()

