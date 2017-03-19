
# coding: utf-8

# In[3]:

import nltk


# In[ ]:

def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document) [1]
    sentences = [nltk.word_tokenize(sent) for sent in sentences] [2]
    sentences = [nltk.pos_tag(sent) for sent in sentences]


# In[ ]:

cp = nltk.RegexpParser('CHUNK: {<V.*> <TO> <V.*>}')
brown = nltk.corpus.brown
for sent in brown.tagged_sents():
    tree = cp.parse(sent)
    for subtree in tree.subtrees():
        if subtree.label() == 'CHUNK': print(subtree)


# In[5]:

sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]

grammar = "NP: {<DT>?<JJ>*<NN>}"

cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)


# In[6]:

type(result)


# In[28]:

for r in result:
    if isinstance(r, nltk.tree.Tree):
        if r.label():
            if r.label() == 'NP':
                print([word for word, tag in r.leaves()])


# In[ ]:



