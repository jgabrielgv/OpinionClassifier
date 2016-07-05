import collections
import nltk
import re
import pickle
import random
from nltk.corpus import stopwords
import pandas as pd
from utility import get_document, open_pickled_file, show_most_informative_features, get_classification
import io

encode = 'latin1'
 
#sys.setdefaultencoding('latin1')

print("Loading documents")
#documents = open_pickled_file("documents.pickle")

encode = 'latin1'

print("Loading words")
#all_words = open_pickled_file("all_words.pickle")

#print("Loading word_features 5k")
#word_features = open_pickled_file("word_features.pickle")
print("Loading word_features all")
word_features = open_pickled_file("word_features_all.pickle")

def find_features(document):
    words = document.lower().strip().split(' ')
    features = {}
    for w in word_features:
        features[w] = (w.lower().decode(encode) in [x.lower().decode(encode) for x in words])

    return features

#print("Loading featuresets5k")
#featuresets = open_pickled_file("featuresets5k.pickle")
print("Loading featuresets_all")
featuresets = open_pickled_file("featuresets_all.pickle")

#amount = len(documents)/10
#training_len = amount*9
#print("Training len", int(training_len))

# set that we'll train our classifier with
training_set = featuresets[:]

# set that we'll test against.
#testing_set = featuresets[:433] + featuresets[866:]

print("Training model with NaiveBayes")
classifier = nltk.NaiveBayesClassifier.train(training_set)

#print("Loading originalnaivebayes5k classifier")
#classifier = open_pickled_file("originalnaivebayes5k.pickle")

#print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

print("Print informative features")
show_most_informative_features(classifier, 50)

#print("End of the process")

def sentiment_file(file_path):
    fo = io.open("result_file.csv", "w+", encoding=encode)
   
    file_text = ""
    document = get_document(file_path)
    for index in range(len(document[0])):
        print("Processing %d of %d..." % (index + 1, len(document[0])))
        text = document[0][index].split(',')[2].strip().lower()
        classification = document[0][index].split(',')[1].strip()

        classified = sentiment(text)
        file_text += "%s,%s\n" % (document[0][index].decode(encode), classified.upper())

    fo.write(file_text);
    fo.close()
    
def sentiment(text):
    feats = find_features(text)
    return classifier.classify(feats)#,(nltk.classify.accuracy(classifier, set(text)))*100

#This is the more pythonic way
#important_words = filter(lambda x: x not in stopwords.words('spanish'), words)
