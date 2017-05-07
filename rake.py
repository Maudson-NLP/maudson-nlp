# Implementation of RAKE - Rapid Automtic Keyword Exraction algorithm
# as described in:
# Rose, S., D. Engel, N. Cramer, and W. Cowley (2010). 
# Automatic keyword extraction from indi-vidual documents. 
# In M. W. Berry and J. Kogan (Eds.), Text Mining: Applications and Theory.unknown: John Wiley and Sons, Ltd.
#
# NOTE: The original code (from https://github.com/aneesha/RAKE)
# has been extended by a_medelyan (zelandiya)
# with a set of heuristics to decide whether a phrase is an acceptable candidate
# as well as the ability to set frequency and phrase length parameters
# important when dealing with longer documents

from __future__ import absolute_import
from __future__ import print_function
import re
import operator
import six
from six.moves import range
from nltk import stem
import json
import numpy as np

debug = False
test = False

def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False


def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    return stop_words


def separate_words(text, min_word_return_size):
    """
    Utility function to return a list of all words that are have a length greater than a specified number of characters.
    @param text The text that must be split in to words.
    @param min_word_return_size The minimum no of characters a word must have to be included.
    """
    splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    words = []
    for single_word in splitter.split(text):
        current_word = single_word.strip().lower()
        #leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
        if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
            words.append(current_word)
    return words


def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    sentence_delimiters = re.compile(u'[\\[\\]\n.!?,;:\t\\-\\"\\(\\)\\\u2019\u2013]')
    sentences = sentence_delimiters.split(text)
    return sentences


def build_stop_word_regex(stop_word_file_path):
    stop_word_list = load_stop_words(stop_word_file_path)
    stop_word_regex_list = []
    for word in stop_word_list:
        word_regex = '\\b' + word + '\\b'
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern



def generate_candidate_keywords(sentence_list, stopword_pattern, min_char_length=1, min_words_length=1, max_words_length=3, min_keyphrase_frequency =1):
    phrase_list = []
    raw_list = []
    for s in sentence_list:
        tmp = re.sub(stopword_pattern, '|', s.strip())
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            raw_list.append(phrase)
    for phraz in raw_list:
        if phraz != "" and is_acceptable(phraz, min_char_length, min_words_length, max_words_length):
                if min_keyphrase_frequency > 1 and raw_list.count(phraz) < min_keyphrase_frequency:
                    continue
                else:
                    phrase_list.append(phraz)
            
    return phrase_list



def is_acceptable(phrase, min_char_length, min_words_length, max_words_length):

    words = phrase.split()
    # a phrase must have a min length in characters
    for word in words: 
        if len(word) < min_char_length:
            return 0

    # a phrase must have a max number of words
    words = phrase.split()
    if len(words) > max_words_length or len(words) < min_words_length:
        return 0

    digits = 0
    alpha = 0
    for i in range(0, len(phrase)):
        if phrase[i].isdigit():
            digits += 1
        elif phrase[i].isalpha():
            alpha += 1

    # a phrase must have at least one alpha character
    if alpha == 0:
        return 0

    # a phrase must have more alpha than digits characters
    if digits > alpha:
        return 0
    return 1



def spell_check(sentence_list):
    
    with open('dico_spell.json', 'r') as f:
        dico_spell = json.load(f)
        
    for i in range(len(sentence_list)):
        sentence_split = sentence_list[i].split(' ')
        for k in range(len(sentence_split)):
            if sentence_split[k] in dico_spell.keys():
                sentence_split[k] = dico_spell[sentence_split[k]]
        sentence_list[i] =  str(' '.join(sentence_split))
    
    return sentence_list



def handle_neg(candidate):
    
    candidate = candidate.lower()
    neg_items = ['not a lot of', 'not', 'no', 'non', 'not', 'nor', 'free of', 'not too', 'not to', 'clear of', 'free of']
    for neg_item in neg_items:
        candidate = candidate.replace(neg_item, 'no')

    word_list = candidate.split(' ')
    if '' in word_list:
        word_list.remove('') 
        
    to_remove = []
    new_phrases = []
    cpt=0
    
    while cpt < len(word_list):
        
        if word_list[cpt][-4:] == 'less':
            new_phrases.append('no_'+word_list[cpt][:-4])
            
            to_remove += [word_list[cpt]]
            cpt += 1
            continue
        
        if cpt+1 < len(word_list):
            if word_list[cpt+1] == 'free':
                new_phrases.append('no_'+word_list[cpt])
                
                to_remove += [word_list[cpt], word_list[cpt+1]]
                cpt += 2
                continue

        if word_list[cpt] == 'no' and cpt+2 < len(word_list):
                                                
            if word_list[cpt+2] == 'or' and cpt+4 < len(word_list):
                if word_list[cpt+4] != 'or':
                    new_phrases.append('no_'+word_list[cpt+1])
                    new_phrases.append('no_'+word_list[cpt+3])
                    
                    to_remove += [word_list[cpt], word_list[cpt+1], word_list[cpt+2], word_list[cpt+3]]
                    cpt += 4
                    continue
                    
                else:
                    new_phrases.append('no_'+word_list[cpt+1])
                    new_phrases.append('no_'+word_list[cpt+3])
                    new_phrases.append('no_'+word_list[cpt+5])
                    
                    to_remove += [word_list[cpt], word_list[cpt+1], word_list[cpt+2], word_list[cpt+3], word_list[cpt+4], word_list[cpt+5]]
                    cpt += 6
                    continue
                    
            elif word_list[cpt+2] == 'or' and cpt+4 >= len(word_list):
                new_phrases.append('no_'+word_list[cpt+1])
                new_phrases.append('no_'+word_list[cpt+3])
                    
                to_remove += [word_list[cpt], word_list[cpt+1], word_list[cpt+2], word_list[cpt+3]]
                cpt += 4
                continue
            
            else:
                new_phrases.append('no_'+word_list[cpt+1])
                
                to_remove += [word_list[cpt], word_list[cpt+1]]
                cpt += 2
                continue
                
        elif word_list[cpt] == 'no' and cpt+1 < len(word_list):
            new_phrases.append('no_'+word_list[cpt+1])
                
            to_remove += [word_list[cpt], word_list[cpt+1]]
            cpt += 2
            continue
            
        else:
            cpt += 1
            continue
    
    to_keep = [el for el in word_list if el not in to_remove]
    new_candidate = to_keep + new_phrases
    
    return ' '.join(new_candidate)



def handle_neg_list(sentence_list):
    sent_handle_neg = []
    for candidate in sentence_list:
        sent_handle_neg.append(handle_neg(candidate))
    return sent_handle_neg



def stem_candidate_keywords(phrase_list):
    
    stemmer = stem.PorterStemmer()
    
    #Stem phrase_list and track
    phrase_list_stem = []
    track_stem = {}
    for phrase in phrase_list:
        spl = phrase.split(' ')
        phrase_stem = ''
        for item in spl:
            if len(phrase_stem)==0:
                phrase_stem += stemmer.stem(item)
            
            else:
                phrase_stem += ' '+stemmer.stem(item)

        phrase_list_stem.append(str(phrase_stem))
        track_stem[phrase] = str(phrase_stem)
    
    #Compute reverse tracking
    track_stem_rev = {}
    for phrase, stem_phrase in track_stem.iteritems():
        if stem_phrase not in track_stem_rev.keys():
            track_stem_rev[stem_phrase] = [phrase]
        else:
            track_stem_rev[stem_phrase].append(phrase)

    #Compute final list of keyphrase candidates
    final_list = list(set(phrase_list_stem))
    final_list = [track_stem_rev[str(phrase)][0] for phrase in final_list]
    
    return final_list, phrase_list_stem, track_stem


def calculate_word_metrics(phrase_list, tradeoff):
    word_frequency = {}
    word_degree = {}
    word_score = {}
    keyphrase_counts = {}

    for phrase in phrase_list:
        keyphrase_counts.setdefault(phrase, 0)
        keyphrase_counts[phrase] += 1
        
        word_list = separate_words(phrase, 0)
        
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            
            word_degree.setdefault(word, 0)
            word_degree[word] += len(word_list) - 1

    for word in word_frequency.keys():
        word_score.setdefault(word, 0)
        if word_degree[word] != 0:
            word_score[word] = word_frequency[word] / np.power(word_degree[word], tradeoff)
        else:
            word_score[word] = word_frequency[word]
    
    keyphrase_stem_freq = {kp: float(sc)/len(phrase_list) for (kp, sc) in keyphrase_counts.iteritems()}
    
    return word_score, keyphrase_stem_freq


def generate_candidate_keyphrase_scores(final_list, word_scores, keyphrase_stem_freq, track_stem):
    keyphrase_candidates = {}
    for phrase in final_list:
        phrase_stem = track_stem[phrase]
        keyphrase_candidates.setdefault(phrase, 0)
        word_list = separate_words(phrase_stem, 0)
        candidate_score = 0
        for word in word_list:
            candidate_score += word_scores[word]
        keyphrase_candidates[phrase] = float(candidate_score) / len(word_list)
        
    keyphrase_candidates = {kp: (sc, keyphrase_stem_freq[track_stem[kp]]) for (kp, sc) in keyphrase_candidates.iteritems()}
    
    return keyphrase_candidates


class Rake(object):
    def __init__(self, stop_words_path, min_char_length=1, min_words_length = 1, max_words_length=3, min_keyphrase_frequency=1, tradeoff=0.8):
        self.__stop_words_path = stop_words_path
        self.__stop_words_pattern = build_stop_word_regex(stop_words_path)
        self.__min_char_length = min_char_length
        self.__min_words_length = min_words_length
        self.__max_words_length = max_words_length
        self.__min_keyphrase_frequency = min_keyphrase_frequency
        self.__tradeoff = tradeoff

    def run(self, text):
        sentence_list = split_sentences(text)

        sentence_list_check = spell_check(sentence_list)
        
        sentence_list_neg_melt = handle_neg_list(sentence_list_check)
        
        phrase_list = generate_candidate_keywords(sentence_list_neg_melt, self.__stop_words_pattern, self.__min_char_length, self.__min_words_length, self.__max_words_length, self.__min_keyphrase_frequency)
                
        final_list, phrase_list_stem, track_stem = stem_candidate_keywords(phrase_list)

        word_scores, keyphrase_stem_freqs = calculate_word_metrics(phrase_list_stem, self.__tradeoff)
        
        keyphrase_candidates = generate_candidate_keyphrase_scores(final_list, word_scores, keyphrase_stem_freqs, track_stem)

        sorted_keywords = sorted(six.iteritems(keyphrase_candidates), key=operator.itemgetter(1), reverse=True)
        
        return sorted_keywords

