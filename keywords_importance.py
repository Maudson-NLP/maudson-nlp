
# coding: utf-8

# In[9]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC


# In[10]:

xl = pd.ExcelFile("./data/Beauty_5.xlsx")


# In[11]:

df = xl.parse()
df = df.dropna()


# In[12]:

df.head()


# In[13]:

x = df.drop(['overall'], axis=1)
y = df['overall']

# Trying first 1000 examples for now
x_train, x_test, y_train, y_test = train_test_split(x[0:10000],
                                                    y[0:10000],
                                                    test_size=0.3)


# In[14]:

def convert_to_numeric(x):
    if type(x) == int:
        return x
    else:
        return 2
    
y_train = y_train.apply(convert_to_numeric)
y_test = y_test.apply(convert_to_numeric)


# In[49]:

# Train the pipeline
vectorizer = CountVectorizer(stop_words='english')
svc = LinearSVC()

text_pipe = make_pipeline(vectorizer, svc)
text_pipe.fit(x_train['reviewText"'], np.array(y_train))


# In[51]:

# Predict some ratings
text_pipe.score(x_test['reviewText"'], y_test)


# In[17]:

# Get top words for classes
def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-20:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))


# In[18]:

print_top10(vectorizer, svc, [0,1,2,3,4])


# In[31]:

def plot_coefficients(classifier, feature_names, top_features=20):
    # Five sets of coefficients correspond to each of the OVR classifiers
    coef = classifier.coef_[4]
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()


# In[32]:

plot_coefficients(svc, vectorizer.get_feature_names())


# In[33]:

print len(vectorizer.vocabulary_)
print len(vectorizer.get_feature_names())
print len(svc.coef_[4])


# In[ ]:



