
# coding: utf-8

# In[36]:

import numpy as np
import pandas as pd

import nltk
from textblob import TextBlob
import RAKE
import textacy

# rake
# textacy
# nltk
# SpaCy
# Gensim
# TextBlob
# CoreNLP


# Pandas indexing note:
    # loc - query by labels
    # iloc - query by indices
    # ix - query by labels, falls back to indices


# In[24]:

survey_data = pd.read_excel('data/survey_data.xlsx')


# # Preprocessing steps
#     - word boundaries
#     - lemmatization
#     - stopwords

# # Textblob

# In[33]:

text = survey_data.iloc[121,3]
print(text)
opinion = TextBlob(text)
opinion.sentiment


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

doc.count('America')
3
bot = doc.to_bag_of_terms(ngrams={2, 3}, as_strings=True)

