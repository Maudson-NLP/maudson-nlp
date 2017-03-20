import os
import pandas as pd

import summarizer



path = os.getcwd() + '/data/test_data/Data/'

bpoil_docs = []
for news_source in os.listdir(path):
    if news_source[:5] == 'bpoil':
        sub_path = path + '/' + news_source + '/InputDocs'
        for news_date in os.listdir(sub_path):
            for article in os.listdir(sub_path + '/' + news_date):
                f = open(sub_path + '/' + news_date + '/' + article)
                bpoil_docs.append(f.read())


print(len(bpoil_docs))

df = pd.DataFrame(bpoil_docs)

print(df.shape)
df.columns = ['BP Oil']
print(df.head())

summary = summarizer.summarize(df, df.columns, use_svd=True, split_longer_sentences=True)

print(summary)
