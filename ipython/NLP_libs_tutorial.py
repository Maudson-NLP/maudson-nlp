
# coding: utf-8

# In[ ]:

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


# # EDA with NLTK

# In[ ]:

nltk.download_shell()


# In[ ]:

from nltk.book import *
from nltk import bigrams


# In[ ]:

print(text1)
print(text1.concordance("monstrous"))
print(text1.similar("monstrous"))
print(text2.common_contexts(["monstrous", "very"]))

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

# text3.generate()

print(sorted(set(text3)))
print(FreqDist(text1))
print(list(bigrams(['more', 'is', 'said', 'than', 'done'])))

fdist = FreqDist(len(w) for w in text1)
fdist.most_common()


# In[ ]:

fdist1 = FreqDist(text1)
fdist1.most_common(10)


# In[ ]:

V = set(text1)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)


# # Keyword Extraction with Rake

# In[26]:

type(text1)


# In[30]:

Rake = RAKE.Rake('SmartStoplist.txt')
Rake.run("Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility "        "of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. "        "Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating"       " sets of solutions for all types of systems are given. These criteria and the corresponding algorithms "        "for constructing a minimal supporting set of solutions can be used in solving all the considered types of "        "systems and systems of mixed types.")


# # Easy analysis with Textblob

# In[31]:

text = survey_data.iloc[121,3]
print(text)
opinion = TextBlob(text)
opinion.sentiment


# In[34]:

from textblob import TextBlob

text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''

blob = TextBlob(text)
print(blob.tags)

print(blob.noun_phrases)

for sentence in blob.sentences:
    print(sentence.sentiment.polarity)

# blob.translate(to="es")  # 'La amenaza titular de The Blob...'


# In[36]:

from textblob import WordList

animals = TextBlob("cat dog octopus")
animals.words

WordList(['cat', 'dog', 'octopus'])
animals.words.pluralize()


# In[37]:

b = TextBlob("I havv goood speling!")
print(b.correct())

from textblob import Word
w = Word('falibility')
w.spellcheck()


# # Diving deeper with SpaCy

# In[43]:

import spacy

nlp = spacy.load('en')
doc = nlp(u'Hello, world. Natural Language Processing in 10 lines of code.')


# ### Get tokens and sentences

# In[44]:

# Get first token of the processed document
token = doc[0]
print(token)

# Print sentences (one sentence per line)
for sent in doc.sents:
    print(sent)


# ### Part of speech tags

# In[45]:

# For each token, print corresponding part of speech tag
for token in doc:
    print('{} - {}'.format(token, token.pos_))


# ### Syntactic dependencies

# In[46]:

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

# In[47]:

# Print all named entities with named entity types
doc_2 = nlp(u"I went to Paris where I met my old friend Jack from uni.")
for ent in doc_2.ents:
    print('{} - {}'.format(ent, ent.label_))


# ### Noun chunks

# In[48]:

# Print noun chunks for doc_2
print([chunk for chunk in doc_2.noun_chunks])


# ### Unigram probabilities

# In[49]:

# For every token in doc_2, print log-probability of the word, estimated from counts from a large corpus 
for token in doc_2:
    print(token, ',', token.prob)


# ### Word embedding / Similarity

# In[50]:

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


# # Diving deeper with Textacy

# ### Efficiently stream documents from disk and into a processed corpus:

# In[52]:

cw = textacy.corpora.CapitolWords()
docs = cw.records(speaker_name={'Hillary Clinton', 'Barack Obama'})
content_stream, metadata_stream = textacy.fileio.split_record_fields(
    docs, 'text')
corpus = textacy.Corpus(u'en', texts=content_stream, metadatas=metadata_stream)
corpus


# ### Represent corpus as a document-term matrix, with flexible weighting and filtering:

# In[53]:

doc_term_matrix, id2term = textacy.vsm.doc_term_matrix(
    (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True)
     for doc in corpus),
    weighting='tfidf', normalize=True, smooth_idf=True, min_df=2, max_df=0.95)
print(repr(doc_term_matrix))


# ### Train and interpret a topic model

# In[54]:

model = textacy.tm.TopicModel('nmf', n_topics=10)
model.fit(doc_term_matrix)
doc_topic_matrix = model.transform(doc_term_matrix)
print(doc_topic_matrix.shape)

for topic_idx, top_terms in model.top_topic_terms(id2term, top_n=10):
    print('topic', topic_idx, ':', '   '.join(top_terms))


# ### Basic indexing as well as flexible selection of documents in a corpus:

# In[55]:

obama_docs = list(corpus.get(lambda doc: doc.metadata['speaker_name'] == 'Barack Obama'))
print(len(obama_docs))
doc = corpus[-1]
print(doc)


# ### Preprocess plain text, or highlight particular terms in it:

# In[56]:

textacy.preprocess_text(doc.text, lowercase=True, no_punct=True)[:70]
textacy.text_utils.keyword_in_context(doc.text, 'America', window_width=35)


# ### Extract various elements of interest from parsed documents:

# In[62]:

print(list(textacy.extract.ngrams(doc, 2, filter_stops=True, filter_punct=True, filter_nums=False))[:15])
print('-'*20)
print(list(textacy.extract.ngrams(doc, 3, filter_stops=True, filter_punct=True, min_freq=2)))
print('-'*20)
print(list(textacy.extract.named_entities(doc, drop_determiners=True, exclude_types=u'numeric'))[:10])
print('-'*20)

pattern = textacy.constants.POS_REGEX_PATTERNS['en']['NP']
print(pattern)
print('-'*20)

print(list(textacy.extract.pos_regex_matches(doc, pattern))[:10])
print('-'*20)
print(list(textacy.extract.semistructured_statements(doc, 'I', cue='be')))
print('-'*20)
print(textacy.keyterms.textrank(doc, n_keyterms=10))


# ### Compute common statistical attributes of a text:

# In[63]:

textacy.text_stats.readability_stats(doc)


# ### Count terms individually, and represent documents as a bag-of-terms with flexible weighting and inclusion criteria:

# In[1]:

print(doc.count('America'))

bot = doc.to_bag_of_terms(ngrams={2, 3}, as_strings=True)
sorted(bot.items(), key=lambda x: x[1], reverse=True)[:10]


# # LDA with Gensim

# In[65]:

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[66]:

from gensim import corpora, models, similarities

corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
    [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
    [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
    [(0, 1.0), (4, 2.0), (7, 1.0)],
    [(3, 1.0), (5, 1.0), (6, 1.0)],
    [(9, 1.0)],
    [(9, 1.0), (10, 1.0)],
    [(9, 1.0), (10, 1.0), (11, 1.0)],
    [(8, 1.0), (10, 1.0), (11, 1.0)]]


# In[67]:

tfidf = models.TfidfModel(corpus)
vec = [(0, 1), (4, 1)]
print(tfidf[vec])

index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)
sims = index[tfidf[vec]]
print(list(enumerate(sims)))


# ### Corpora and Vector Spaces

# In[70]:

from gensim import corpora

documents = ["Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"]


# In[71]:

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
    for document in documents]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

from pprint import pprint  # pretty-printer
pprint(texts)


# In[72]:

dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference
print(dictionary)


# In[73]:

print(dictionary.token2id)


# In[75]:

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored


# In[76]:

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
print(corpus)


# ### Corpus Streaming – One Document at a Time

# In[115]:

class MyCorpus(object):
    def __iter__(self):
        for line in open('mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            print(line)
            yield dictionary.doc2bow(line.lower().split())


# In[116]:

corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
print(corpus_memory_friendly)


# In[117]:

for vector in corpus_memory_friendly:  # load one vector into memory at a time
    print(vector)


# In[87]:

from six import iteritems
# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))
# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)


# ### Corpus Formats

# In[88]:

# create a toy corpus of 2 documents, as a plain Python list
corpus = [[(1, 0.5)], []]  # make one document empty, for the heck of it

corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)


# In[89]:

corpora.SvmLightCorpus.serialize('/tmp/corpus.svmlight', corpus)
corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)
corpora.LowCorpus.serialize('/tmp/corpus.low', corpus)


# In[90]:

corpus = corpora.MmCorpus('/tmp/corpus.mm')


# In[91]:

print(corpus)


# In[92]:

# one way of printing a corpus: load it entirely into memory
print(list(corpus))  # calling list() will convert any sequence to a plain Python list


# In[93]:

# another way of doing it: print one document at a time, making use of the streaming interface
for doc in corpus:
    print(doc)


# In[94]:

corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)


# ### Compatibility with NumPy and SciPy

# In[99]:

import gensim
import numpy as np

numpy_matrix = np.random.randint(10, size=[5,2])  # random matrix as an example
corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
gensim.matutils.corpus2dense(corpus, num_terms=10)


# In[98]:

import scipy.sparse

scipy_sparse_matrix = scipy.sparse.random(5,2)  # random sparse matrix as example
corpus = gensim.matutils.Sparse2Corpus(scipy_sparse_matrix)
gensim.matutils.corpus2csc(corpus)


# ### Topics and Transformations

# In[101]:

import os
from gensim import corpora, models, similarities

if (os.path.exists("/tmp/deerwester.dict")):
   dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
   corpus = corpora.MmCorpus('/tmp/deerwester.mm')
   print("Used files generated from first tutorial")
else:
   print("Please run first tutorial to generate data set")


# In[102]:

tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model


# In[103]:

doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow]) # step 2 -- use the model to transform vectors


# In[104]:

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)


# In[105]:

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi


# In[106]:

lsi.print_topics(2)


# In[107]:

for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print(doc)


# In[108]:

lsi.save('/tmp/model.lsi') # same for tfidf, lda, ...
lsi = models.LsiModel.load('/tmp/model.lsi')


# ### Available transformations

# In[109]:

model = models.TfidfModel(corpus, normalize=True)


# In[111]:

model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)


# In[112]:

model.add_documents(another_tfidf_corpus) # now LSI has been trained on tfidf_corpus + another_tfidf_corpus
lsi_vec = model[tfidf_vec] # convert some new document into the LSI space, without affecting the model

model.add_documents(more_documents) # tfidf_corpus + another_tfidf_corpus + more_documents
lsi_vec = model[tfidf_vec]


# In[114]:

model = models.RpModel(corpus_tfidf, num_topics=500)
model = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
model = models.HdpModel(corpus, id2word=dictionary)


# ### Similarity Queries

# In[118]:

from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
corpus = corpora.MmCorpus('/tmp/deerwester.mm') # comes from the first tutorial, "From strings to vectors"
print(corpus)

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print(vec_lsi)

index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it
index.save('/tmp/deerwester.index')
index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
sims = index[vec_lsi] # perform a similarity query against the corpus
print(list(enumerate(sims))) # print (document_number, document_similarity) 2-tuples

sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims) # print sorted (document number, similarity score) 2-tuples

'''
[(2, 0.99844527), # The EPS user interface management system
(0, 0.99809301), # Human machine interface for lab abc computer applications
(3, 0.9865886), # System and human system engineering testing of EPS
(1, 0.93748635), # A survey of user opinion of computer system response time
(4, 0.90755945), # Relation of user perceived response time to error measurement
(8, 0.050041795), # Graph minors A survey
(7, -0.098794639), # Graph minors IV Widths of trees and well quasi ordering
(6, -0.1063926), # The intersection graph of paths in trees
(5, -0.12416792)] # The generation of random binary unordered trees


# In[ ]:



