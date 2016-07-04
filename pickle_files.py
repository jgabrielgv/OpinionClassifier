import collections
import nltk
import re
import pickle
import random
from nltk.corpus import stopwords
import pandas as pd
 
#sys.setdefaultencoding('latin1')

def open_pickled_file(file_name):
    open_file = open("pickled_algos/%s" % (file_name), "rb")
    item = pickle.load(open_file)
    open_file.close()

    return item

def pickle_file(file_name, item):
    save_item = open(file_name,"wb")
    pickle.dump(item, save_item)
    save_item.close()

def get_document(file_name):
    return pd.read_csv(file_name, header=None, \
                    delimiter="\t", quoting=3)

print("Begin loading document")
trainning_file = get_document("Trainning.csv")
#remove punctuation and split into seperate words
print("End loading document")

def get_classification(classification):
    polarity = ""
    if(classification == "NONE"):
        polarity = "NONE"
    else:
        polarity = "OPINION"
    return polarity

def load_documents():
    for index in range(len(trainning_file[0])):
        text = trainning_file[0][index].split(',')[2].strip().lower()
        classification = trainning_file[0][index].split(',')[1].strip()
        documents.append((text.split(' '), get_classification(classification)))

documents = []

print("Creating documents")
load_documents()
#documents = open_pickled_file("documents.pickle")

print("Creating pickled documents")
pickle_file("pickled_algos/documents.pickle", documents)

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
pickle_file("pickled_algos/all_words.pickle", all_words)

word_features = list(all_words.keys())[:5000]

print("Creating pickled word_features")
pickle_file("pickled_algos/word_features.pickle", word_features)

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
#featuresets = open_pickled_file("featuresets5k.pickle")

print("Creating pickled featuresets5k")
pickle_file("pickled_algos/featuresets5k.pickle", featuresets)

amount = len(documents)/10
training_len = amount*9
print("Training len", int(training_len))

# set that we'll train our classifier with
training_set = featuresets[:training_len]

# set that we'll test against.
testing_set = featuresets[training_len:]

#print("Features")
#print(find_features(["gusta", ":-))"]))

print("Training model with NaiveBayes")
classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Creating pickled NaiveBayes classifier")
pickle_file("pickled_algos/originalnaivebayes5k.pickle", classifier)

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

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

print("End of the process")
