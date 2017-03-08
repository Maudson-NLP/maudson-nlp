import nltk
import scipy
import gensim
import sklearn
import numpy as np
import pandas as pd
import logging

from pprint import pprint

import scipy.sparse
from gensim import corpora
import sklearn.metrics.pairwise
from gensim.models import Doc2Vec
from collections import defaultdict
from scipy.sparse import csr_matrix
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



def make_sentences(df):
    # todo - automate for each column
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    textBlobs = []
    for i in range(2 ,3):
        if str(df.ix[: ,i].dtype) == 'object':
            # Use a period as most users do not end their answers with periods
            # (Possible source of optimization)
            textBlob = df.ix[: ,i].str.cat(sep='. ')
            tokenized = tokenizer.tokenize(textBlob)
            textBlobs += tokenized

    return np.array(textBlobs)


def do_stemming(sentences):
    stemmer = PorterStemmer()

    stemmed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)

        stemmed_sentence = []
        for w in words:
            stemmed_sentence.append(stemmer.stem(w))
        stemmed_sentences.append(' '.join(stemmed_sentence))

    return stemmed_sentences


# Todo - only removes first stopword of bigram
def remove_stopword_bigrams(sentences):
    sw_bigrams_removed = []
    for sentence in sentences:
        bigrams = nltk.bigrams(sentence.split())
        stopwords = nltk.corpus.stopwords.words('english')

        filtered = [tup for tup in bigrams if not [True for wrd in tup if wrd in stopwords].count(True) == 2]
        # Join back into sentence:
        joined = " ".join("%s" % tup[0] for tup in filtered)
        sw_bigrams_removed.append(joined)

    return sw_bigrams_removed


def vectorize(sentences):
    return CountVectorizer().fit_transform(sentences)


def ortho_proj_vec(vectors, B):
    print("Calculating vector with largest distance to subspace of {} basis vectors".format(len(B)))
    projs = csr_matrix(vectors.shape, dtype=np.int8) # coo_matrix

    iteration = 0
    for b in B:
        iteration += 1
        print("Starting with basis vector {} of {}".format(iteration, len(B)))

        p_i = np.multiply(vectors.dot(b.T), b)
        projs += p_i

    dists = scipy.sparse.linalg.norm((vectors - projs), axis=1)

    print("Top distance: {}".format(np.max(dists)))
    print("And its index: {}".format(np.argmax(dists)))
    return np.argmax(dists)


def sem_vol_max(sentences, vectors, L):
    S = set()
    B = set()

    # Mean vector
    c = (1 / vectors.shape[0]) * np.sum(vectors, axis=0)
    c = csr_matrix(c)

    # 1st furthest vector -- Will this always just be the longest sentence?
    dists = sklearn.metrics.pairwise.pairwise_distances(vectors, c)
    p = np.argmax(dists)
    print("Sentence furthest from mean: {}".format(sentences[p]))
    S.add(sentences[p])

    # 2nd furthest vector
    dists = sklearn.metrics.pairwise.pairwise_distances(vectors, vectors[p])
    q = np.argmax(dists)
    print("Sentence furthest from first: {}".format(sentences[q]))
    S.add(sentences[q])

    b_0 = vectors[q] / scipy.sparse.linalg.norm(vectors[q])
    B.add(b_0)

    total_length = len(sentences[p].split()) + len(sentences[q].split())
    exceeded_length_count = 0

    for i in range(0, vectors.shape[0]):
        print("\n" + "*"* 10)

        r = ortho_proj_vec(vectors, B)
        print("Furthest sentence: " + sentences[r])
        print("Total words: {}".format(total_length))
        print("Length of sentence to add: {}".format(len(sentences[r].split())))

        new_sentence_length = len(sentences[r].split())

        if total_length + new_sentence_length <= L:
            S.add(sentences[r])
            b_r = np.divide(vectors[r], scipy.sparse.linalg.norm(vectors[r]))
            B.add(b_r)
            total_length = total_length + new_sentence_length
            # Reset the exceeded_length_count
            exceeded_length_count = 0
        else:
            print("Sentence too long to add to set")
            # Temporary hack to prevent us from choosing this vector again:
            vectors[r] = np.zeros(vectors[p].shape)

            exceeded_length_count += 1
            if exceeded_length_count >= 15:
                break

    print("Final sentence count: " + str(len(S)))
    return S


def summarize(filename):
    xl = pd.ExcelFile(filename)
    df = xl.parse()
    df = df.dropna()

    L = 90
    delimiter = '\n' + '*' * 30

    print(delimiter + ' Raw sentences:')
    sentences = make_sentences(df)
    print(sentences[:10])

    print(delimiter + ' After stemming:')
    stemmed = do_stemming(sentences)
    print(stemmed[:10])

    print(delimiter + ' After removing stopword bigrams:')
    sw_bigrams_removed = remove_stopword_bigrams(stemmed)
    print(sw_bigrams_removed[:2])

    print(delimiter + ' After vectorization:')
    vectorized = vectorize(sw_bigrams_removed)
    print(vectorized[:2])

    print(delimiter + ' Run Algorithm:')
    summary = sem_vol_max(sentences, vectorized, L)

    print(delimiter + ' Result:')
    print(summary)
    return repr(summary)
