import scipy
import logging
import sklearn
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.metrics.pairwise
from scipy.sparse import csr_matrix
from preprocessing import *


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DELIMITER = '\n' + '*' * 30 + ' '


def ortho_proj_vec(vectors, B):
    """
    Find the vector that is furthest from the subspace spanned by the vectors in B
    :param vectors: List of vectors to search
    :param B: Set of unit vectors that form the basis of the subspace
    :return:
    """
    print(DELIMITER + "Calculating vector with largest distance to subspace of {} basis vectors".format(len(B)))
    projs = csr_matrix(vectors.shape, dtype=np.int8)  # try coo_matrix?

    iteration = 0
    for b in B:
        iteration += 1
        # print("Starting with basis vector {} of {}".format(iteration, len(B)))
        p_i = np.multiply(vectors.dot(b.T), b)
        projs += p_i

    dists = scipy.sparse.linalg.norm((vectors - projs), axis=1)

    print("Top distance: {}".format(np.max(dists)))
    print("And its index: {}".format(np.argmax(dists)))
    return np.argmax(dists)


def compute_mean_vector(vectors):
    c = (1 / vectors.shape[0]) * np.sum(vectors, axis=0)
    return csr_matrix(c)


def compute_primary_basis_vector(vectors, sentences, d, L):
    """
    Find the vector in vectors with furthest distances from the given vector d, while the
    length of the corresponding sentence from sentences does not overflow the word limit L
    :param vectors: List of vectorized sentences to search through
    :param sentences: Corresponding list of sentences
    :param d: Test vector to compare to each vector in vectors
    :param L: Max length of word count in sentence
    :return: Index of vector with largest distance in vectors
    """
    dists = sklearn.metrics.pairwise.pairwise_distances(vectors, d)
    p = np.argmax(dists)
    # Skip vectors that overflow the word limit
    total_length = len(sentences[p].split())
    # Include length of first vector if we aren't dealing with `d` as the mean vector. Todo

    while total_length > L:
        print("Basis vector too long, recalculating...")
        vectors[p] = np.zeros(vectors[p].shape)
        dists = sklearn.metrics.pairwise.pairwise_distances(vectors, d)
        p = np.argmax(dists)
        total_length = len(sentences[p].split())
        if type(d) != scipy.sparse.csr.csr_matrix:
            total_length += len(sentences[d].split())

    return p


def sem_vol_max(sentences, vectors, L):
    """
    Perform Semantic Volume Maximization as specified in Yogotama et al.
    :param sentences: List of original, un-preprocessed sentences
    :param vectors: np.array of vectorized sentences corresponding to sentences
    :param L: Limit of number of words to include in summary
    :return:
    """
    S = set()
    B = set()

    # Mean vector
    c = compute_mean_vector(vectors)

    # 1st furthest vector -- Will this always just be the longest sentence?
    p = compute_primary_basis_vector(vectors, sentences, c, L)
    vec_p = vectors[p]
    sent_p = sentences[p]
    print("Sentence furthest from mean: {}".format(sent_p))
    S.add(sent_p)

    # 2nd furthest vector
    q = compute_primary_basis_vector(vectors, sentences, vec_p, L)
    vec_q = vectors[q]
    sent_q = sentences[q]
    print("Sentence furthest from the first: {}".format(sent_q))
    S.add(sent_q)

    b_0 = vec_q / scipy.sparse.linalg.norm(vec_q)
    B.add(b_0)

    return sentence_add_loop(vectors, sentences, S, B, L)


def sentence_add_loop(vectors, sentences, S, B, L):
    """
    Iteratively add most distance vector in `vectors`, from list of vectors `vectors`, to set B.
    Add the corresponding sentences from `sentences` to set S.
    :param vectors: Vectors to search through
    :param sentences: Corresponding sentences to add to set
    :param S: Set of sentences to be returned
    :param B: Set of existing basis vectors of the subspace to be compared to each new vector
    :param L: Max length of total number of words across all sentences in S
    :return: List of sentences corresponding to set of vectors in `vectors`
    that maximally span the subspace of `vectors`
    """
    total_length = sum([len(sent.split()) for sent in S])
    exceeded_length_count = 0

    for i in range(0, vectors.shape[0]):
        r = ortho_proj_vec(vectors, B)
        print(DELIMITER)
        print("Furthest sentence: " + sentences[r])
        print("Total words: {}".format(total_length))
        print("Length of sentence to add: {}".format(len(sentences[r].split())))

        new_sentence_length = len(sentences[r].split())

        if total_length + new_sentence_length <= L:
            S.add(sentences[r])
            b_r = np.divide(vectors[r], scipy.sparse.linalg.norm(vectors[r]))
            B.add(b_r)
            total_length += new_sentence_length
            # Reset the exceeded_length_count
            exceeded_length_count = 0
            # Prevent us from adding this sentence again
            # Todo - original authors had this same problem?
            vectors[r] = np.zeros(vectors[r].shape)
        else:
            print("Sentence too long to add to set")
            # Temporary hack to prevent us from choosing this vector again:
            vectors[r] = np.zeros(vectors[r].shape)

            exceeded_length_count += 1
            if exceeded_length_count >= 15:
                break

    print("Final sentence count: " + str(len(S)))
    return [str(e) for e in S]


def summarize(filename, columns, l=100, use_bigrams=False, use_svd=False, k=100, use_noun_phrases=False):
    """
    Start summarization task on excel file with columns to summarize
    :param filename: String - Name of excel file with columns to summarize
    :param columns: List<String> - Titles in first row of spreadsheet for columns to summarize
    :param l: Integer - Max length of summary (in words)
    :param use_bigrams: Boolean - Whether to summarize based on word counts or bigrams
    :param use_svd: Boolean - Whether to summarize based on top k word concepts
    :param k: Integer - Number of top word concepts to incorporate
    :return: List of summary strings
    """
    data_dir = './uploaded_data/'
    xl = pd.ExcelFile(data_dir + filename)
    df = xl.parse()
    df = df.dropna()

    print(DELIMITER + 'Raw sentences:')
    sentence_sets = make_sentences(df, columns)
    print(sentence_sets[0][:10])

    summaries = []
    # Now we iterate over sentence groups for each column and summarize each
    for sentence_set in sentence_sets:
        print(DELIMITER + 'After lemmatization:')
        lemmatized = do_lemmatization(sentence_set)
        print(lemmatized[:10])

        # Todo - very slow as compared to in python...
        # print(delimiter + ' After spellcheck:')
        # spellchecked = do_spellcheck(lemmatized)
        # print(spellchecked[:10])

        print(DELIMITER + 'After removing stopword bigrams:')
        sw_bigrams_removed = remove_stopword_bigrams(lemmatized)
        print(sw_bigrams_removed[:2])
        print(len(sw_bigrams_removed))

        if use_bigrams:
            print(DELIMITER + 'After vectorization (no bigrams):')
            vectorized = vectorize_bigrams(sw_bigrams_removed)
            print(vectorized[:2])
        else:
            print(DELIMITER + 'After vectorization (using bigrams):')
            vectorized = vectorize(sw_bigrams_removed)
            print(vectorized[:2])

        if use_svd:
            print(DELIMITER + 'After SVD:')
            vectorized = vectorized.asfptype()
            U, s, V = scipy.sparse.linalg.svds(vectorized, k=k)
            print("U: {}, s: {}, V: {}".format(U.shape, s.shape, V.shape))
            vectorized = csr_matrix(U)

        print(DELIMITER + 'Run Algorithm:')
        summary = sem_vol_max(sentence_set, vectorized, l)

        print(DELIMITER + 'Result:')
        print(summary)

        # Optionally include a list of noun phrases
        if use_noun_phrases:
            summaries.append((summary, extract_noun_phrases(sentence_set)))
        else:
            summaries.append((summary, []))

    print("Columns summarized: {}".format(len(summaries)))
    return summaries
