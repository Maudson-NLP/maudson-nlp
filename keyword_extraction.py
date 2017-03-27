from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.insert(0, 'kw_include')
import pandas as pd
import rake


def extract_keyphrases(filename, nb_kp, min_char_length, max_words_length, min_keyword_frequency):
    
    nb_kp = int(nb_kp)
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
        int1 (1): min_char_length, minimum number of characters per word
        int2 (5): max_words_length, maximum number of words per keyphrase
        int4 (1): min_keyword_frequency, minimum nb of occurences for the keyword in the text.
        '''
    rake_object = rake.Rake(stoppath, int(min_char_length), int(max_words_length), int(min_keyword_frequency))
    keyphrases = {qst_clean[k]: rake_object.run(srv_concat[k])[:nb_kp] for k in range(len(qst_clean))}
    keyphrases = {qst:[(kw, "%.2f" % ct) for (kw,ct) in lst] for (qst,lst) in keyphrases.iteritems()}
    
    return keyphrases




