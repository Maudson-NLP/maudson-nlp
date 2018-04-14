
# coding: utf-8

# In[16]:

import rake
import pandas as pd
import numpy as np
import re


# In[20]:

filename = '../data/survey_unilever.xlsx'
nb_kp = 10
stopword_pattern = rake.build_stop_word_regex("SmartStoplist.txt")
min_char_length=1
max_words_length=3
headers=''
groupby=''


# In[21]:

nb_kp = float(nb_kp)
srv_raw = pd.ExcelFile(filename)
srv = srv_raw.parse()

if len(headers) == 0:
    del srv[list(srv)[0]]
    qst = list(srv)
    qst = [str(q).split("\n")[0] for q in list(srv)]
    srv.columns = qst
else:
    qst = headers.split('%')
    cols = [q.strip('\n') for q in srv.columns]
    srv.columns = cols

srv_concat = [reduce(lambda x,y: str(x)+'. '+str(y), srv[qst[k]]) for k in range(len(qst))]


# In[22]:

text = srv_concat[0]


# In[23]:

sentence_list = rake.split_sentences(text)

sentence_list_check = rake.spell_check(sentence_list)
        
sentence_list = rake.handle_neg_list(sentence_list_check)


# In[31]:

min_keyphrase_frequency=2
phrase_list = []
raw_list = []
for s in sentence_list[:10]:
    tmp = re.sub(stopword_pattern, '|', s.strip())
    phrases = tmp.split("|")
    
    for phrase in phrases:
        phrase = phrase.strip().lower()
        raw_list.append(phrase)
print(raw_list)
for phraz in raw_list:
    if phraz != "" and rake.is_acceptable(phraz, min_char_length, max_words_length):
        if min_keyphrase_frequency > 1 and raw_list.count(phraz) < min_keyphrase_frequency:
            continue
        else:
            phrase_list.append(phraz)


# In[32]:

phrase_list


# In[ ]:



