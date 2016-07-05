import collections
import nltk
import re
import pickle
import random
from nltk.corpus import stopwords
import pandas as pd
from utility import get_document, pickle_file, open_pickled_file, show_most_informative_features, get_classification

print("Begin loading document")
trainning_file = get_document("Trainning.csv")
#remove punctuation and split into seperate words
print("End loading document")

def load_documents():
    for index in range(len(trainning_file[0])):
        text = trainning_file[0][index].split(',')[2].strip().lower()
        classification = trainning_file[0][index].split(',')[1].strip()
        documents.append((text.split(' '), get_classification(classification)))

documents = []

print("Creating documents")
load_documents()
documents = open_pickled_file("documents.pickle")

print("Creating pickled documents")
pickle_file("Pickle/documents.pickle", documents)

encode = 'latin1'

# use these three lines to do the replacement

def load_words():
    for document in documents:
        for word in document[0]:
            word = word.strip()
            if len(word) > 1 and word.decode(encode) not in stopwords.words('spanish'):
            #if word not in stopwords.words('spanish') and len(word) > 1:
                all_words.append(word)

all_words = []

print("Creating all words")
load_words()
#all_words = open_pickled_file("all_words.pickle")

print("Defining frequences")
all_words = nltk.FreqDist(all_words)

print("Creating pickled words")
pickle_file("Pickle/all_words_all.pickle", all_words)

word_features = list(all_words.keys())
#word_features = open_pickled_file("word_features_all.pickle")

print("Creating pickled word_features")
pickle_file("Pickle/word_features_all.pickle", word_features)

#featuresets = [(find_features(rev), category) for (rev, category) in documents[0]]
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w.decode(encode) in [x.decode(encode) for x in words])
        #features[w] = (w in words)

    return features

def add_features():
    for document in documents:
        featuresets.append((find_features(document[0]), document[1]))

print("Creating feature sets")
featuresets = []
add_features()
#featuresets = open_pickled_file("featuresets_all.pickle")

print("Creating pickled featuresets_all")
pickle_file("Pickle/featuresets_all.pickle", featuresets)

amount = len(documents)/10
training_len = amount*9
#print("Training len", int(training_len))

# set that we'll train our classifier with
#training_set = featuresets[:training_len]

# set that we'll test against.
testing_set = featuresets[:]

print("Training model with NaiveBayes")
classifier = nltk.NaiveBayesClassifier.train(featuresets)
#classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Creating pickled NaiveBayes classifier")
pickle_file("Pickle/originalnaivebayes_all.pickle", classifier)

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

print("Print informative features")
show_most_informative_features(classifier, 200)

print("End of the process")
