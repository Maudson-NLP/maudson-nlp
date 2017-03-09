
# coding: utf-8

# In[36]:

import numpy as np
import pandas as pd

import nltk
from textblob import TextBlob
import RAKE
import textacy

# nltk
# rake
# TextBlob
# SpaCy
# Textacy
# Gensim
# CoreNLP


# Pandas indexing note:
    # loc - query by labels
    # iloc - query by indices
    # ix - query by labels, falls back to indices


# In[24]:

survey_data = pd.read_excel('data/survey_data.xlsx')


# # NLTK

# In[ ]:

nltk.download()
from nltk.book import *
print(text1)
print(text1.concordance("monstrous"))
print(text1.similar("monstrous"))
print(text2.common_contexts(["monstrous", "very"]))

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

text3.generate()

print(sorted(set(text3)))

print(FreqDist(text1))

print(list(bigrams(['more', 'is', 'said', 'than', 'done'])))

fdist = FreqDist(len(w) for w in text1)
fdist.most_common()


# # Rake

# In[ ]:

Rake = RAKE.Rake(['SmartStoplist.txt'])
Rake.run(text)


# # Textblob

# text = survey_data.iloc[121,3]
# print(text)
# opinion = TextBlob(text)
# opinion.sentiment

# # SpaCy

# In[ ]:

nlp = spacy.load('en')
doc = nlp(u'Hello, world. Natural Language Processing in 10 lines of code.')


# ### Get tokens and sentences

# In[ ]:

# Get first token of the processed document
token = doc[0]
print(token)

# Print sentences (one sentence per line)
for sent in doc.sents:
    print(sent)


# ### Part of speech tags

# In[ ]:

# For each token, print corresponding part of speech tag
for token in doc:
    print('{} - {}'.format(token, token.pos_))


# ### Syntactic dependencies

# In[ ]:

# Write a function that walks up the syntactic tree of the given token and collects all tokens to the root token (including root token).

def tokens_to_root(token):
    """
    Walk up the syntactic tree, collecting tokens to the root of the given `token`.
    :param token: Spacy token
    :return: list of Spacy tokens
    """
    tokens_to_r = []
    while token.head is not token:
        tokens_to_r.append(token)
        token = token.head
        tokens_to_r.append(token)

    return tokens_to_r

# For every token in document, print it's tokens to the root
for token in doc:
    print('{} --> {}'.format(token, tokens_to_root(token)))

# Print dependency labels of the tokens
for token in doc:
    print('-> '.join(['{}-{}'.format(dependent_token, dependent_token.dep_) for dependent_token in tokens_to_root(token)]))


# ### Named entities

# In[ ]:

# Print all named entities with named entity types
doc_2 = nlp(u"I went to Paris where I met my old friend Jack from uni.")
for ent in doc_2.ents:
    print('{} - {}'.format(ent, ent.label_))


# ### Noun chunks

# In[ ]:

# Print noun chunks for doc_2
print([chunk for chunk in doc_2.noun_chunks])


# ### Unigram probabilities

# In[ ]:

# For every token in doc_2, print log-probability of the word, estimated from counts from a large corpus 
for token in doc_2:
    print(token, ',', token.prob)


# ### Word embedding / Similarity

# In[ ]:

# For a given document, calculate similarity between 'apples' and 'oranges' and 'boots' and 'hippos'
doc = nlp(u"Apples and oranges are similar. Boots and hippos aren't.")
apples = doc[0]
oranges = doc[2]
boots = doc[6]
hippos = doc[8]
print(apples.similarity(oranges))
print(boots.similarity(hippos))

print()
# Print similarity between sentence and word 'fruit'
apples_sent, boots_sent = doc.sents
fruit = doc.vocab[u'fruit']
print(apples_sent.similarity(fruit))
print(boots_sent.similarity(fruit))


# # Textacy

# ### Efficiently stream documents from disk and into a processed corpus:

# In[ ]:

cw = textacy.corpora.CapitolWords()
docs = cw.records(speaker_name={'Hillary Clinton', 'Barack Obama'})
content_stream, metadata_stream = textacy.fileio.split_record_fields(
    docs, 'text')
corpus = textacy.Corpus('en', texts=content_stream, metadatas=metadata_stream)
corpus


# ### Represent corpus as a document-term matrix, with flexible weighting and filtering:

# In[ ]:

doc_term_matrix, id2term = textacy.vsm.doc_term_matrix(
    (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True)
     for doc in corpus),
    weighting='tfidf', normalize=True, smooth_idf=True, min_df=2, max_df=0.95)
print(repr(doc_term_matrix))


# ### Train and interpret a model

# In[ ]:

model = textacy.tm.TopicModel('nmf', n_topics=10)
model.fit(doc_term_matrix)
doc_topic_matrix = model.transform(doc_term_matrix)
print(doc_topic_matrix.shape)

for topic_idx, top_terms in model.top_topic_terms(id2term, top_n=10):
    print('topic', topic_idx, ':', '   '.join(top_terms))


# ### Basic indexing as well as flexible selection of documents in a corpus:

# In[ ]:

obama_docs = list(corpus.get(lambda doc: doc.metadata['speaker_name'] == 'Barack Obama'))
print(len(obama_docs))
doc = corpus[-1]
print(doc)


# ### Preprocess plain text, or highlight particular terms in it:

# In[ ]:

textacy.preprocess_text(doc.text, lowercase=True, no_punct=True)[:70]
textacy.text_utils.keyword_in_context(doc.text, 'America', window_width=35)


# ### Extract various elements of interest from parsed documents:

# In[ ]:

# print(list(textacy.extract.ngrams(doc, 2, filter_stops=True, filter_punct=True, filter_nums=False))[:15])

# print(list(textacy.extract.ngrams(doc, 3, filter_stops=True, filter_punct=True, min_freq=2)))

# print(list(textacy.extract.named_entities(doc, drop_determiners=True, exclude_types='numeric'))[:10])

pattern = textacy.constants.POS_REGEX_PATTERNS['en']['NP']
# print(pattern)

# print(list(textacy.extract.pos_regex_matches(doc, pattern))[:10])

# print(list(textacy.extract.semistructured_statements(doc, 'I', cue='be')))

# print(textacy.keyterms.textrank(doc, n_keyterms=10))


# ### Compute common statistical attributes of a text:

# In[ ]:

textacy.text_stats.readability_stats(doc)


# ### Count terms individually, and represent documents as a bag-of-terms with flexible weighting and inclusion criteria:

# In[ ]:

print(doc.count('America'))

bot = doc.to_bag_of_terms(ngrams={2, 3}, as_strings=True)
sorted(bot.items(), key=lambda x: x[1], reverse=True)[:10]


# # Gensim

# In[ ]:



