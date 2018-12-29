import csv
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.snowball import EnglishStemmer
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

from svm_classifier.es_classifier.es_a_classifier_1 import tokenize, normalizer_listing, normalizer

stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()
stopwords = set(stopwords.words("english"))

def readfile(path):
    tweets = []
    hs = []
    tr = []
    ag = []
    id = []
    with open(path, encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            tweets.append(line[1])
            id.append(line[0])
            if(line[2]=='HS'):
                hs.append(line[2])
            else:
                hs.append(int(float(line[2])))
            if (line[3] == 'TR'):
                tr.append(line[3])
            else:
                tr.append(int(float(line[3])))
            if (line[4] == 'AG'):
                ag.append(line[4])
            else:
                ag.append(int(float(line[4])))

        id.pop(0)
        tweets.pop(0)
        hs.pop(0)
        tr.pop(0)
        ag.pop(0)
    return id, tweets, hs, tr, ag



def stem(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    return result


def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr


if __name__ == "__main__":

    # Read spanish (train + trial) data
    id, tweets, hate_speech, target, aggressive = readfile('../../datasets/public_development_en/train_en.tsv')
    trial_data = readfile('../../datasets/trial_en.tsv')
    id = id + trial_data[0]
    tweets = tweets + trial_data[1]
    hate_speech = hate_speech + trial_data[2]
    target = target + trial_data[3]
    aggressive = aggressive + trial_data[4]

    # Create DataFrame with data
    t = pd.DataFrame()
    t['text'] = tweets
    t['hate_decision'] = hate_speech
    t['target'] = target
    t['aggressive'] = aggressive
    t['normalized_tweet'] = t.text.apply(normalizer_listing)

    # Split Data into two parts : train and test
    train, test = train_test_split(t, test_size=0.25, random_state=10)
    X_train = train['text'].values
    X_test = test['text'].values
    y_bit1_train = train['hate_decision']
    y_bit1_test = test['hate_decision']
    y_bit2_train = train['target']
    y_bit2_test = test['target']
    y_bit3_train = train['aggressive']
    y_bit3_test = test['aggressive']

    # Prepare features
    vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=normalizer_listing,
        lowercase=True,
        ngram_range=(1, 3),
        stop_words=stopwords)


    # Provides train/test indices to split data in train/test sets
    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    np.random.seed(1)

    # Create the pipeline SVM
    pipeline_svm = make_pipeline(vectorizer,SVC(probability=True, kernel="linear", class_weight="balanced", max_iter=-1))

    # Gris search : find the best parameters to have the highest f1 score
    grid_svm_one = GridSearchCV(pipeline_svm,
                             param_grid={'svc__C': [0.01, 0.025, 0.05, 0.1, 1, 10, 100]},
                             cv=kfolds,
                             scoring="f1",
                             verbose=1,
                             n_jobs=-1)
    grid_svm_two = GridSearchCV(pipeline_svm,
                                param_grid={'svc__C': [0.01, 0.025, 0.05, 0.1, 1, 10, 100]},
                                cv=kfolds,
                                scoring="f1",
                                verbose=1,
                                n_jobs=-1)
    grid_svm_three = GridSearchCV(pipeline_svm,
                                param_grid={'svc__C': [0.01, 0.025, 0.05, 0.1, 1, 10, 100]},
                                cv=kfolds,
                                scoring="f1",
                                verbose=1,
                                n_jobs=-1)

    # fit the  svm model
    grid_svm_one.fit(X_train, y_bit1_train)
    grid_svm_two.fit(X_train, y_bit2_train)
    grid_svm_three.fit(X_train, y_bit3_train)

    # Start prediction
    f = open("../svm_predictions/en_b.tsv", "w",encoding="utf8")
    with open('../../datasets/public_development_en/dev_en.tsv', encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line_number, line in enumerate(tsvreader):
            if(line_number>0):
                bit_one = grid_svm_one.predict([normalizer(line[1])])
                if bit_one[0] == 0 :
                    f.write(str(line[0]) + "\t"+ "0"+ "\t"+"0" + "\t"+"0")
                else:
                    bit_two = grid_svm_two.predict([normalizer(line[1])])
                    bit_three = grid_svm_three.predict([normalizer(line[1])])
                    f.write(str(line[0])+"\t")
                    f.write(str(bit_one[0])+"\t")
                    f.write(str(bit_two[0]) + "\t")
                    f.write(str(bit_three[0]) + "\t")
                f.write('\n')
    f.close()





