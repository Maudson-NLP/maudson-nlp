import scipy
import logging
import sklearn
import pandas as pd
import scipy.sparse
import sklearn.metrics.pairwise
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import csr_matrix
from preprocessing import *


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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

    iteration = 0
    for b in B:
        iteration += 1
        p_i = np.multiply(vectors.dot(b.T), b)
        projs += p_i

    dists = scipy.sparse.linalg.norm((vectors - projs), axis=1)

    print("Top distance: {}".format(np.max(dists)))
    print("And its index: {}".format(np.argmax(dists)))
    return np.argmax(dists)


def compute_mean_vector(vectors):
    c = np.mean(vectors, axis=0)
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
    dists = sklearn.metrics.pairwise.pairwise_distances(vectors, d) # should be proj_b0^ui ?
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


def summarize(data,
              columns=[],
              group_by=None,
              l=100,
              ngram_range=(1,1),
              use_svd=False,
              k=100,
              scale_vectors=False,
              use_noun_phrases=False,
              extract_sibling_sents=False,
              split_longer_sentences=False,
              to_split_length=50):
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
    :return: List of summary strings
    """
    if type(data) == str or type(data) == unicode:
        data_dir = './uploaded_data/'
        xl = pd.ExcelFile(data_dir + data)
        df = xl.parse()
        df = df.dropna()
    else:
        df = data

    print(DELIMITER + 'Raw sentences:')
    if group_by:
        # todo - only one columm supported for grouped summarization
        sentence_sets = make_sentences_by_group(df, group_by, columns[0])
    else:
        sentence_sets = make_sentences_from_dataframe(df, columns)

    summaries = []
    # Now we iterate over sentence groups for each column and summarize each
    for i, sentence_set in enumerate(sentence_sets):

        # todo - ugly
        if group_by:
            group = sentence_set[0]
            sentence_set = sentence_set[1]

        if split_longer_sentences:
            sentence_set = split_long_sentences(sentence_set, to_split_length)
        if extract_sibling_sents:
            sentence_set = extract_sibling_sentences(sentence_set)

        vectors = do_lemmatization(sentence_set)
        vectors = remove_stopword_bigrams(vectors)

        vectors = vectorize(vectors, ngram_range=ngram_range)

        if scale_vectors:
            scaler = MaxAbsScaler()
            vectors = scaler.fit_transform(vectors)

        if use_svd:
            use_alt_k = False
            if k > min(vectors.shape):
                use_alt_k = True
                k_new = min(vectors.shape) - 1

                vectors = vectors.asfptype()
            if use_alt_k:
                U, s, V = scipy.sparse.linalg.svds(vectors, k=k_new)
            else:
                U, s, V = scipy.sparse.linalg.svds(vectors, k=k)

            print(DELIMITER + 'After SVD:')
            print("U: {}, s: {}, V: {}".format(U.shape, s.shape, V.shape))
            vectors = csr_matrix(U)


        print(DELIMITER + 'Run Algorithm:')
        summary = sem_vol_max(sentence_set, vectors, l)

        print(DELIMITER + 'Result:')
        print(summary)

        # Optionally include a list of noun phrases
        if group_by:
            toappend = [group, summary, []]
        else:
            current_column = columns[i] if len(columns) else df.columns[i]
            toappend = [current_column, summary, []]

        # todo - ugly
        if use_noun_phrases:
            toappend[2] = extract_noun_phrases(sentence_set)

        summaries.append(toappend)

    print("Columns summarized: {}".format(len(summaries)))
    return summaries
