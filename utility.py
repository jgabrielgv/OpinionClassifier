import pickle
import pandas as pd

encode = "latin1"

def open_pickled_file(file_name):
    open_file = open("Pickles/%s" % (file_name), "rb")
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

def get_classification(classification):
    polarity = ""
    if(classification == "NONE"):
        polarity = "NONE"
    else:
        polarity = "OPINION"
    return polarity
