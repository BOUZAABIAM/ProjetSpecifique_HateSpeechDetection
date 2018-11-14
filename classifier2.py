import csv

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.snowball import EnglishStemmer
from classifier import tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from classifier import normalizer_listing

stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()
en_stopwords = set(stopwords.words("english"))

def readfile(path):
    tweets = []
    hs = []
    tr = []
    ag = []
    with open(path, encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            tweets.append(line[1])
            if(line[2]=='HS'):
                hs.append(line[2])
            else:
                hs.append(int(float(line[2])))
            tr.append(line[3])
            ag.append(line[4])
        tweets.pop(0)
        hs.pop(0)
        tr.pop(0)
        ag.pop(0)
    return tweets, hs, tr, ag



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
    tweets, hate_speech, target, aggressive = readfile('./datasets/trial_en.tsv')
    t = pd.DataFrame()
    t['text'] = tweets
    t['hate_decision'] = hate_speech

    t['normalized_tweet'] = t.text.apply(normalizer_listing)
    train, test = train_test_split(t, test_size=0.4, random_state=10)
    X_train = train['text'].values
    X_test = test['text'].values
    y_train = train['hate_decision']
    y_test = test['hate_decision']

    vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=normalizer_listing,
        lowercase=True,
        ngram_range=(1, 3),
        stop_words=en_stopwords)

    '''Provides train/test indices to split data in train/test sets'''
    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

    np.random.seed(1)
    pipeline_svm = make_pipeline(vectorizer,SVC(probability=True, kernel="linear", class_weight="balanced"))
    grid_svm = GridSearchCV(pipeline_svm,
                             param_grid={'svc__C': [0.01, 0.1, 1, 10, 100]},
                             cv=kfolds,
                             scoring="roc_auc",
                             verbose=0,
                             n_jobs=-1)

    grid_svm.fit(X_train, y_train)
    #print(grid_svm.score(X_test, y_test))
    print(grid_svm.best_params_)
    print(grid_svm.best_score_)

    print(report_results(grid_svm.best_estimator_, X_test, y_test))

    roc_svm = get_roc_curve(grid_svm.best_estimator_, X_test, y_test)
    fpr, tpr = roc_svm
    plt.figure(figsize=(14, 8))
    plt.plot(fpr, tpr, color="red")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve : Receiver Operating Characteristic')
    plt.show()

