###################################################
# Implementation of Semantic Volume Maximization
# as specified by:
# Yogatama, Dani et al. 2015.
# Extractive Summarization by Maximizing Semantic Volume.
# Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing: 1961-1966. September, 2015.
###################################################

import os
# import boto
# from boto.s3.key import Key
import json
import xlrd
import scipy
import logging
import sklearn
import pandas as pd
import scipy.sparse
import sklearn.metrics.pairwise
from sklearn.preprocessing import Normalizer
from scipy.sparse import csr_matrix
from preprocessing import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Uncomment for AWS S3 functionality
# keyId = os.environ['S3_KEY']
# sKeyId= os.environ['S3_SECRET']
# conn = boto.connect_s3(keyId, sKeyId)

DELIMITER = '\n' + '*' * 30 + ' '



def ortho_proj_vec(vectors, B):
    """
    Find the vector that is furthest from the subspace spanned by the vectors in B
    :param vectors: List of vectors to search
    :param B: Set of unit vectors that form the basis of the subspace
    :return: Index of furthest vector
    """
    print(DELIMITER + "Calculating vector with largest distance to subspace of {} basis vectors".format(len(B)))
    projs = csr_matrix(vectors.shape, dtype=np.int8)  # try coo_matrix?

    for b in B:
        p_i = np.multiply(vectors.dot(b.T), b)
        projs += p_i

    dists = scipy.sparse.linalg.norm((vectors - projs), axis=1)

    print("Top distance: {}".format(np.max(dists)))
    print("And its index: {}".format(np.argmax(dists)))
    return np.argmax(dists)


def compute_mean_vector(vectors):
    c = vectors.mean(axis=0)
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
    # Todo: should be proj_b0^ui ?
    dists = sklearn.metrics.pairwise.pairwise_distances(vectors, d)
    p = np.argmax(dists)

    # Skip vectors that overflow the word limit
    total_length = len(sentences[p].split())

    # Todo: Include length of first vector if we aren't dealing with `d` as the mean vector
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

        # Todo - norm may be zero if sentence only had stopwords
        norm = scipy.sparse.linalg.norm(vectors[r])

        if total_length + new_sentence_length <= L and norm != 0:
            b_r = np.divide(vectors[r], norm)

            S.add(sentences[r])
            B.add(b_r)

            total_length += new_sentence_length
            # Reset the exceeded_length_count

            exceeded_length_count = 0
            # Prevent us from adding this sentence again
            # Todo - original authors had this same problem?
            vectors[r] = np.zeros(vectors[r].shape)

        else:
            print("Sentence too long to add to set, or sentence consists only of stopwords")
            # Temporary hack to prevent us from choosing this vector again:
            vectors[r] = np.zeros(vectors[r].shape)

            exceeded_length_count += 1
            if exceeded_length_count >= 15:
                break

    print("Final sentence count: " + str(len(S)))
    return [str(e) for e in S]


def get_summary_for_sentence_set(sentence_set, column, **args):
    if args['split_longer_sentences']:
        sentence_set = split_long_sentences(sentence_set, args['to_split_length'])
    if args['exclude_misspelled']:
        sentence_set = do_exclude_misspelled(sentence_set)
    if args['extract_sibling_sents']:
        sentence_set = extract_sibling_sentences(sentence_set)

    vectors = do_lemmatization(sentence_set)
    vectors = remove_stopword_bigrams(vectors)
    vectors = vectorize(vectors, ngram_range=args['ngram_range'], tfidf=args['tfidf'])

    if args['scale_vectors']:
        normalizer = Normalizer()
        vectors = normalizer.fit_transform(vectors)

    if args['use_svd']:
        vectors = vectors.asfptype()

    print(vectors.shape)
    print('k value; ' + str(args['k']))
    if args['k'] >= min(vectors.shape):
        print("k too large for vectors shape, lowering...")
        args['k'] = min(vectors.shape) - 1
    if args['k'] == 0:
        # Very repetitive sentences
        return [column, ['Summary not possible'], []]

    if args['use_svd']:
        U, s, V = scipy.sparse.linalg.svds(vectors, k=args['k'])
        print(DELIMITER + 'After SVD:')
        print("U: {}, s: {}, V: {}".format(U.shape, s.shape, V.shape))
        vectors = csr_matrix(U)

    print(DELIMITER + 'Run Algorithm:')
    summary = sem_vol_max(sentence_set, vectors, args['l'])

    print(DELIMITER + 'Result:')
    print(summary)

    result = [column, summary, []]

    if args['use_noun_phrases']:
        # Todo - Use struct for results
        result[2] = extract_noun_phrases(sentence_set)

    return result


def summarize(
    id,
    data,
    columns=[],
    group_by=None,
    l=100,
    ngram_range=(2,3),
    tfidf=False,
    use_svd=False,
    k=100,
    scale_vectors=False,
    use_noun_phrases=False,
    extract_sibling_sents=False,
    split_longer_sentences=False,
    to_split_length=50,
    exclude_misspelled=False):
    """
    Start summarization task on excel file with columns to summarize
    :param data: String - Name of excel file with columns to summarize
        or pandas.DataFrame - DataFrame to summarize by column
    :param columns: List<String> - Titles in first row of spreadsheet for columns to summarize
    :param l: Integer - Max length of summary (in words)
    :param use_bigrams: Boolean - Whether to summarize based on word counts or bigrams
    :param use_svd: Boolean - Whether to summarize based on top k word concepts
    :param k: Integer - Number of top word concepts to incorporate
    :param extract_sibling_sents: Boolean - whether to split sentences into individual siblings as defined by adjacent S tags in nltk parse tree
    :param split_long_sentences: Boolean - whether to split longer sentences into shorter ones to prevent bias in distance calculation
    :param to_split_length: Integer - Length above which to split sentences
    :return: List of lists of summary sentences
    """
    # Todo: use **kwargs
    args = {
        'split_longer_sentences': split_longer_sentences,
        'exclude_misspelled': exclude_misspelled,
        'extract_sibling_sents': extract_sibling_sents,
        'ngram_range': ngram_range,
        'scale_vectors': scale_vectors,
        'use_svd': use_svd,
        'columns': columns,
        'use_noun_phrases': use_noun_phrases,
        'tfidf': tfidf,
        'k': k,
        'l': l,
    }

    upload_folder = os.path.join(os.getcwd(), 'uploaded_data')

    if type(data) == str or type(data) == unicode:
        # AWS S3 functionality
        # bucketName = "clever-nlp"
        # bucket = conn.get_bucket(bucketName)
        # key = Key(bucket, data)
        # Get the contents of the key into a file
        # key.get_contents_to_filename(data)

        # Todo - messy...
        # It is either an excel file or plain text
        # Using a bad sep so that it's all just one cell
        try:
            xl = pd.ExcelFile(os.path.join(upload_folder, data))
            df = xl.parse()
        except xlrd.biffh.XLRDError as e:
            print(e)
            print(type(e))
            print(e.message)
            df = pd.read_csv(os.path.join(upload_folder, data), header=None, error_bad_lines=False, encoding='utf8')

        # df = df.dropna(axis=0, how='all')
        # df = df.dropna(axis=1, how='all')
        # df = df.dropna()
    else:
        df = data

    print(DELIMITER + 'Raw sentences:')
    if group_by:
        sentence_sets, columns = make_sentences_by_group(df, group_by, columns[0])
    else:
        sentence_sets, columns = make_sentences_from_dataframe(df, columns)

    summaries = []
    # Iterate over sentence groups for each column and summarize each
    for i, sentence_set in enumerate(sentence_sets):
        summary = get_summary_for_sentence_set(sentence_set, columns[i], **args)
        summaries.append(summary)

    print("Columns summarized: {}".format(len(summaries)))

    output_folder = 'summary_results'
    filename = id + '.json'
    with open(os.path.join(output_folder, filename), 'w') as outfile:
        json.dump(summaries, outfile)

    # Note: Uncomment for AWS S3 functionality
    # Write result to S3
    #path = './summary_results/'
    # file = open(filename)
    # key.key = filename
    # Upload the file
    # result contains the size of the file uploaded
    # result = key.set_contents_from_file(file)

    return summaries
