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
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



def make_sentences(df, columns):
    '''
    Concatenate columns of data frame into list of lists of sentences
    :param df: Pandas DataFrame
    :return: list of lists of sentences
    '''
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentence_sets = []
    for col in columns:
        if str(df[col].dtype) == 'object':
            # Use a period as most users do not end their answers with periods
            # (Possible source of optimization)
            textBlob = df[col].str.cat(sep='. ')
            tokenized = tokenizer.tokenize(textBlob)
            sentence_sets.append(tokenized)

    return np.array(sentence_sets)


def do_stemming(sentences):
    '''
    Stem sentences using Porter stemmer
    :param sentences: list of sentences to stem
    :return: list of stemmed sentences
    '''
    stemmer = PorterStemmer()

    stemmed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)

        stemmed_sentence = []
        for w in words:
            stemmed_sentence.append(stemmer.stem(w))
        stemmed_sentences.append(' '.join(stemmed_sentence))

    return stemmed_sentences


def do_lemmatization(sentences):
    '''
    Run NLTK lemmatization on sentences
    :param sentences: List of sentences
    :return: List of lemmatized sentences
    '''
    wlem = WordNetLemmatizer()

    lemma_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)

        lemma_sentence = []
        for w in words:
            lemma_sentence.append(wlem.lemmatize(w))
        lemma_sentences.append(' '.join(lemma_sentence))

    return lemma_sentences


# Todo - only removes first stopword of bigram
def remove_stopword_bigrams(sentences):
    '''
    Removes bigrams that consist of only stopwords, as suggested by Yogotama et al
    :param sentences: list of sentences
    :return: list of sentences with stopword bigrams removed
    '''
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
    '''
    Vectorize sentences
    :param sentences: list of sentences
    :return: vectorized word counts. N (# sentences) x M (length of dictionary)
    '''
    return CountVectorizer().fit_transform(sentences)


def ortho_proj_vec(vectors, B):
    '''
    Find the vector that is furthest from the subspace spanned by the vectors in B
    :param vectors: List of vectors to search
    :param B: Set of unit vectors that form the basis of the subspace
    :return:
    '''
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
    '''
    Perform Semantic Volume Maximization as specified in Yogotama et al.
    :param sentences: List of original, un-preprocessed sentences
    :param vectors: np.array of vectorized sentences corresponding to sentences
    :param L: Limit of number of words to include in summary
    :return:
    '''
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
    return [str(e) for e in S]


def summarize(filename, columns, l=100):
    '''
    Start summarization task on excel file with columns to summarize
    :param filename: Name of excel file with columns to summarize
    :param columns: Titles in first row of spreadsheet for columns to summarize
    :param l: Max length of summary (in words)
    :return: Summary string
    '''
    data_dir = './uploaded_data/'
    xl = pd.ExcelFile(data_dir + filename)
    df = xl.parse()
    df = df.dropna()

    delimiter = '\n' + '*' * 30

    print(delimiter + ' Raw sentences:')
    sentence_sets = make_sentences(df, columns)
    print(sentence_sets[0][:10])

    summaries = []
    for sentence_set in sentence_sets:
        print(delimiter + ' After lemmatization:')
        lemmatized = do_lemmatization(sentence_set)
        print(lemmatized[:10])

        print(delimiter + ' After removing stopword bigrams:')
        sw_bigrams_removed = remove_stopword_bigrams(lemmatized)
        print(sw_bigrams_removed[:2])

        print(delimiter + ' After vectorization:')
        vectorized = vectorize(sw_bigrams_removed)
        print(vectorized[:2])

        print(delimiter + ' Run Algorithm:')
        summary = sem_vol_max(sentence_set, vectorized, l)

        print(delimiter + ' Result:')
        print(summary)
        summaries.append(summary)

    print("Columns summarized: {}".format(len(summaries)))
    print(summaries)
    return summaries
