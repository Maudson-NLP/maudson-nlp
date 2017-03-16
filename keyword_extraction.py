from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.insert(0, 'kw_include')
import pandas as pd
import rake


def extract_keyphrases(filename, lgt = 10, min_char_length=5, max_words_length=3, min_keyword_frequency=4):
   
    srv_raw = pd.ExcelFile(filename)
    srv = srv_raw.parse()
    del srv[list(srv)[0]]
    qst = list(srv)
    qst_clean = [str(q).split("\n")[0] for q in list(srv)]
    
    srv_concat = [reduce(lambda x,y: str(x)+'. '+str(y), srv[qst[k]]) for k in range(len(srv.columns))]
    
    stoppath = "kw_include/SmartStoplist.txt"
    '''
    Arguments: 
        stoppath: path to the list of stopwords
        int1 (5): min_char_length, minimum number of characters per word
        int2 (3): max_words_length, maximum number of words per keyphrase
        int4 (4): min_keyword_frequency, minimum nb of occurences for the keyword in the text.
    '''
    rake_object = rake.Rake(stoppath, min_char_length, max_words_length, min_keyword_frequency)
    
    keywords = {qst_clean[k]: rake_object.run(srv_concat[k])[:lgt] for k in range(len(qst_clean))}
    
    return keywords




