import numpy as np
import nltk
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import FeatureHasher


def make_sentences(df, columns):
    """
    Concatenate columns of data frame into list of lists of sentences
    :param df: Pandas DataFrame
    :param columns: list of strings of columns to convert to documents
    :return: list of lists of sentences
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # Strip whitespace from columns - todo: just use indices instead?
    df.columns = [col.strip() for col in df.columns]
    sentence_sets = []
    for col in columns:
        if str(df[col].dtype) == 'object':
            # Use a period as most users do not end their answers with periods
            # todo - Possible source of optimization
            text_blob = df[col].str.cat(sep='. ')
            tokenized = tokenizer.tokenize(text_blob)
            sentence_sets.append(tokenized)

    return np.array(sentence_sets)


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

    return sw_bigrams_removed


def vectorize(sentences):
    """
    Vectorize sentences using plain sklearn.feature_extraction.CountVectorizer.
    Represents the corpus as a matrix of sentences by word counts.
    :param sentences: list of sentences
    :return: scipy.sparse.coo_matrix - vectorized word counts. N (# sentences) x M (length of vocabulary)
    """
    vectorizer = CountVectorizer().fit(sentences)
    return vectorizer.transform(sentences)


def vectorize_bigrams(sentences):
    """
    Vectorize sentences using sklearn.feature_extraction.FeatureHasher.
    Represents the corpus as a matrix of sentences by unique bigram count.
    :param sentences:
    :return:
    """
    fh = FeatureHasher(input_type='string')
    bigrams = [nltk.bigrams(sentence.split()) for sentence in sentences]
    vectorized = fh.transform(((' '.join(x) for x in sample) for sample in bigrams))
    return vectorized


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