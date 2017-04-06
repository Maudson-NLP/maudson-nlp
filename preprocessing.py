import numpy as np
import nltk
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


DELIMITER = '\n' + '*' * 30 + ' '


def make_sentences_from_dataframe(df, columns):
    """
    Concatenate columns of data frame into list of lists of sentences
    :param df: pd.DataFrame
    :param columns: list of strings of columns to convert to documents
    :return: list of lists of sentences
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # Strip whitespace from columns - todo: just use indices instead?
    if len(columns) == 0:
        # Just use all columns
        df.columns = [col.strip() for col in df.columns]
        columns = df.columns
    sentence_sets = []
    for col in columns:
        if str(df[col].dtype) == 'object':
            # Use a period as some users do not end their answers with periods
            # todo - but a lot of people do...
            text_blob = df[col].str.cat(sep='. ')
            tokenized = tokenizer.tokenize(text_blob)
            sentence_sets.append(tokenized)

    print(sentence_sets[0][:10])
    return np.array(sentence_sets)


def make_sentences_by_group(df, group_by_col, column):
    """
    Concatenate columns of dataframe into list of sentences, by group.
    :param df: pd.DataFrame
    :param column: String - column which contains text to concatenate for each group
    :param group_by_col: String - column on which to group
    :return: List<List<String>> - List of lists of sentences
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # grouped = df.groupby(group_by_col, as_index=False)

    sentence_sets = []
    for group in df[group_by_col].unique():
        sentences = df[df[group_by_col] == group][column].str.cat(sep='. ')
        tokenized = tokenizer.tokenize(sentences)
        sentence_sets.append((group, tokenized))

    print(sentence_sets[0][:10])
    return sentence_sets


def split_long_sentences(sentences, l):
    """
    Split longer sentences so that they don't always take precedence in vector distance computation.
    Todo - worthy of optimization
    :param: sentences - list of sentences
    :param: l - if sentence word count is longer than l, split into two sentences at l
    :return: List of split sentences, which will be longer than the input sentences list
    """
    sentence_split_list = []
    for sentence in sentences:
        sentence_split = sentence.split()
        if len(sentence_split) > l:
            first_half = ' '.join(sentence_split[:l])
            second_half = ' '.join(sentence_split[l:])
            sentence_split_list.append(first_half)
            sentence_split_list.append(second_half)
        else:
            sentence_split_list.append(sentence)

    print(DELIMITER + 'After sentence splitting:')
    print(sentence_split_list[:10])
    return sentence_split_list


def extract_sibling_sentences(sentences):
    '''
    Parse sentence POS tree and extract a list of noun phrases
    :param sentences: List<String> of sentences to extract from
    :return: List<String> of all noun phrases
    '''
    # todo: copypasta - cleanme

    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    grammar = "NP: {<DT>?<JJ>*<NN>}"

    cp = nltk.RegexpParser(grammar)
    result = [cp.parse(sentence) for sentence in sentences]

    sibling_sentences = []
    for s in result:
        if isinstance(s, nltk.tree.Tree):
            if s.label():
                if s.label() == 'S':
                    nps = [word for word, tag in s.leaves()]
                    sibling_sentences.append(' '.join(nps))

    print(DELIMITER + 'After sibling sentences extraction:')
    print(sibling_sentences[:10])
    return sibling_sentences


def do_spellcheck(sentences):
    sentences_spellchecked = []
    for sentence in sentences:
        b = TextBlob(sentence)
        sentences_spellchecked.append(b.correct())
    return sentences_spellchecked


def do_stemming(sentences):
    """
    Stem sentences using Porter stemmer
    :param sentences: list of sentences to stem
    :return: list of stemmed sentences
    """
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
    """
    Run NLTK lemmatization on sentences
    :param sentences: List of sentences
    :return: List of lemmatized sentences
    """
    wlem = WordNetLemmatizer()

    lemma_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)

        lemma_sentence = []
        for w in words:
            lemma_sentence.append(wlem.lemmatize(w))
        lemma_sentences.append(' '.join(lemma_sentence))

    print(DELIMITER + 'After lemmatization:')
    print(lemma_sentences[:10])
    return lemma_sentences


# Todo - only removes first stopword of bigram
def remove_stopword_bigrams(sentences):
    """
    Removes bigrams that consist of only stopwords, as suggested by Yogotama et al
    :param sentences: list of sentences
    :return: list of sentences with stopword bigrams removed
    """
    sw_bigrams_removed = []
    for sentence in sentences:
        bigrams = nltk.bigrams(sentence.split())
        stopwords = nltk.corpus.stopwords.words('english')

        filtered = [tup for tup in bigrams if not [True for wrd in tup if wrd in stopwords].count(True) == 2]
        # Join back into sentence:
        joined = " ".join("%s" % tup[0] for tup in filtered)
        sw_bigrams_removed.append(joined)

    print(DELIMITER + 'After removing stopword bigrams:')
    print(sw_bigrams_removed[:2])
    print(len(sw_bigrams_removed))
    return sw_bigrams_removed


def vectorize(sentences, ngram_range=(1,1), tfidf=False):
    """
    Vectorize sentences using plain sklearn.feature_extraction.CountVectorizer.
    Represents the corpus as a matrix of sentences by word counts.
    :param sentences: list of sentences
    :return: scipy.sparse.coo_matrix - vectorized word counts. N (# sentences) x M (length of vocabulary)
    """
    if tfidf:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    else:
        vectorizer = CountVectorizer(ngram_range=ngram_range)
    vectorizer.fit(sentences)

    transformed = vectorizer.transform(sentences)
    print(DELIMITER + 'After vectorization (ngram_range: {}):'.format(ngram_range))
    print(transformed.shape)
    print(transformed[:3])

    return transformed


def extract_noun_phrases(sentences):
    '''
    Parse sentence POS tree and extract a list of noun phrases
    :param sentences: List<String> of sentences to extract from
    :return: List<String> of all noun phrases
    '''
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    grammar = "NP: {<DT>?<JJ>*<NN>}"

    cp = nltk.RegexpParser(grammar)
    result = [cp.parse(sentence) for sentence in sentences]

    noun_phrases = []
    for s in result:
        for r in s:
            if isinstance(r, nltk.tree.Tree):
                if r.label():
                    if r.label() == 'NP':
                        nps = [word for word, tag in r.leaves()]
                        noun_phrases.append(' '.join(nps))

    return noun_phrases