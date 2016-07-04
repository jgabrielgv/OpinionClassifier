import collections
import nltk
import re
import pickle
import random
from nltk.corpus import stopwords
import pandas as pd

encode = 'latin1'
 
#sys.setdefaultencoding('latin1')

def open_pickled_file(file_name):
    open_file = open("pickled_algos/%s" % (file_name), "rb")
    item = pickle.load(open_file)
    open_file.close()

    return item

print("Loading documents")
documents = open_pickled_file("documents.pickle")

encode = 'latin1'

print("Loading words")
all_words = open_pickled_file("all_words.pickle")

print("Loading word_features")
word_features = open_pickled_file("word_features.pickle")

def find_features(document):
    words = document.lower().strip().split(' ')
    features = {}
    for w in word_features:
        features[w] = (w.lower().decode(encode) in [x.lower().decode(encode) for x in words])
        #features[w] = (w in words)

    return features

print("Loading featuresets5k")
#featuresets = open_pickled_file("featuresets5k.pickle")

print("Print feature sets")
#print(featuresets)

amount = len(documents)/10
training_len = amount*9
print("Training len", int(training_len))

# set that we'll train our classifier with
#training_set = featuresets[433:866]

# set that we'll test against.
#testing_set = featuresets[:433] + featuresets[866:]

#print("Training model with NaiveBayes")
#classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Loading originalnaivebayes5k classifier")
classifier = open_pickled_file("originalnaivebayes5k.pickle")

#print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

def show_most_informative_features(classifier, n=10):
        # Determine the most relevant features, and display them.
        cpdist = classifier._feature_probdist
        print('Most Informative Features')

        for (fname, fval) in classifier.most_informative_features(n):
            def labelprob(l):
                return cpdist[l, fname].prob(fval)

            labels = sorted([l for l in classifier._labels
                             if fval in cpdist[l, fname].samples()],
                            key=labelprob)
            if len(labels) == 1:
                continue
            l0 = labels[0]
            l1 = labels[-1]
            if cpdist[l0, fname].prob(fval) == 0:
                ratio = 'INF'
            else:
                ratio = '%8.1f' % (cpdist[l1, fname].prob(fval) /
                                   cpdist[l0, fname].prob(fval))
            print(('%24s = %-14r %6s : %-6s = %s : 1.0' %
                   (fname.decode(encode), fval, ("%s" % l1.decode(encode))[:6].decode(encode), ("%s" % l0.decode(encode))[:6].decode(encode), ratio.decode(encode))))

print("Print informative features")
show_most_informative_features(classifier, 50)

#print("End of the process")

def sentiment(text):
    feats = find_features(text)
    return classifier.classify(feats)#,(nltk.classify.accuracy(classifier, set(text)))*100

#This is the more pythonic way
#important_words = filter(lambda x: x not in stopwords.words('spanish'), words)
