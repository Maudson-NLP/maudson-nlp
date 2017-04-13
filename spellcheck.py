# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 23:33:54 2017

@author: julmaud
"""

import pandas as pd
import json
import enchant
from nltk.tokenize import RegexpTokenizer


data = pd.ExcelFile('uploaded_data/survey_data.xlsx')
txt = data.parse()
cols = list(txt.columns)
cols = cols[1:]
text = ''

#Concat text
for col in cols:
    for k in range(len(txt[col])):
        text += str(txt[col][k])+'. '

#Tokenize & filter
toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
txt_lst = toker.tokenize(text)
txt_lst = [el.lower() for el in txt_lst if len(el) > 2]
txt_lst = list(set(txt_lst))

#Only keep misspeled words
d = enchant.Dict("en_US")
txt_bad = [el for el in txt_lst if d.check(el) == False]

dico_spell = {}
k = 0

while k < len(txt_bad):
    correc = raw_input('Correction of '+txt_bad[k]+' : ')
    if correc == 'goback':
        k -= 1
    else:
        dico_spell[txt_bad[k]]=correc
        k += 1

with open('dico_spell.json', 'w') as fp:
    json.dump(dico_spell, fp, sort_keys = True)