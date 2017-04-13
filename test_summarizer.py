import os
import json
import pandas as pd

import summarizer


path = os.getcwd() + '/data/test_data/Data/'

topics = [
	'bpoil',
	'EgyptianProtest',
	'Finan',
	'H1N1',
	'haiti',
	'IraqWar',
	'LibyaWar',
	'MJ',
	'SyrianCrisis',
]

topics_docs = {
	'bpoil': [],
	'EgyptianProtest': [],
	'Finan': [],
	'H1N1': [],
	'haiti': [],
	'IraqWar': [],
	'LibyaWar': [],
	'MJ': [],
	'SyrianCrisis': [],
}

def filter_dsstore(arr):
	return filter(lambda x: not '.DS' in x, arr)

for topic_folder in filter_dsstore(os.listdir(path)):
	topic = filter(lambda x: x in topic_folder, topics)[0]

	sub_path = path + '/' + topic_folder + '/InputDocs'
	for news_date in filter_dsstore(os.listdir(sub_path)):
		for article in filter_dsstore(os.listdir(sub_path + '/' + news_date)):
			f = open(sub_path + '/' + news_date + '/' + article)
			topics_docs[topic].append(f.read())


df = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in topics_docs.iteritems() ]))

print(df[:2])
summary = summarizer.summarize(
	data=df,
	columns=df.columns,
    l=50,
    tfidf=True,
    ngram_range=(2,3),
    use_svd=True,
    k=100,
    scale_vectors=True,
    use_noun_phrases=False,
    split_longer_sentences=False,
    extract_sibling_sents=True
)

with open('test_result.txt', 'w') as outfile:
    json.dump(summary, outfile)

print(summary)

