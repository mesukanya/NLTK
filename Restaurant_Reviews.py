import numpy as np
import pandas as pd
# library to clean data
import re
# Natural Language Tool Kit
import nltk
nltk.download('stopwords')
# to remove stopword
from nltk.corpus import stopwords
# for Stemming propose
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
# Splitting the dataset into
# the Training set and Test set
from sklearn.model_selection import train_test_split

# Fitting Random Forest Classification
# to the Training set
from sklearn.ensemble import RandomForestClassifier

# Import dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')

# Initialize empty array
# to append clean text
corpus = []

# 1000 (reviews) rows to clean
for i in range(0, 1000):
    # column : "Review", row ith
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    print(review)
    # convert all cases to lower cases
    review = review.lower()
    print(review)
    # split to array(default delimiter is " ")
    review = review.split()
    print(review)
    # creating PorterStemmer object to
    # take main stem of each word
    ps = PorterStemmer()

    # loop for stemming each word
    # in string array at ith row
    review = [ps.stem(word) for word in review
              if not word in set(stopwords.words('english'))]
    print(review)
    # rejoin all string array elements
    # to create back into a string
    review = ' '.join(review)
    print(review)
    # append each string to create
    # array of clean text
    corpus.append(review)
    print(corpus)
    #print(corpus)
    # To extract max 1500 feature.
    # "max_features" is attribute to
    # experiment with to get better results
    cv = CountVectorizer(max_features=1500)

    # X contains corpus (dependent variable)
    X = cv.fit_transform(corpus).toarray()
    print(X)
    # y contains answers if review
    # is positive or negative
    y = dataset.iloc[:, 1].values

    # experiment with "test_size"
    # to get better results
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # n_estimators can be said as number of
    # trees, experiment with n_estimators
    # to get better results
    model = RandomForestClassifier(n_estimators=501,
                                   criterion='entropy')
    model.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = model.predict(X_test)

    print(y_pred)


