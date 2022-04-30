#importing required libraries: numpy, pandas, itertools and some sklearn functions and methods
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

#reading .csv file
df = pd.read_csv('news.csv')

#reading shape and head of table
df.shape
df.head()

#getting labels of the table
labels = df.label
labels.head()

#splitting dataset to training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size = 0.2, random_state = 7)

#initializing tfidf_Vectorizer
#filters according to stop_words and turns collection of raw documents into matrix of TD-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

#fitting and transforming sets
tfidf_train= tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

#initializing PassiveAggressiveClassifier algorithm to have max iteration of 50
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)       #fitting training sets

#predicting test set and calculating accuracy score
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

#creating confusion matrix for false negatives and positives matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])