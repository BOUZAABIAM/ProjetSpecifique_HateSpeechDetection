'''
    SVM + sentiment analysis + cross validation + word and char ngram (stacking)
'''

import csv

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.snowball import EnglishStemmer
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from svm_classifier.en_classifier.en_a_classifier_1 import normalizer, tokenize

stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()
stopwords = set(stopwords.words("spanish"))

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
            tr.append(line[3])
            ag.append(line[4])
        id.pop(0)
        tweets.pop(0)
        hs.pop(0)
        tr.pop(0)
        ag.pop(0)
    return id, tweets, hs, tr, ag


if __name__ == "__main__":
    id, tweets, hate_speech, target, aggressive = readfile('../../datasets/public_development_en/train_en.tsv')
    trial_data = readfile('../../datasets/trial_en.tsv')
    id = id+trial_data[0]
    tweets = tweets + trial_data[1]
    hate_speech = hate_speech + trial_data[2]
    target = target + trial_data[3]

    train = pd.DataFrame()
    train['text'] = tweets
    train['hate_decision'] = hate_speech
    train['normalized_tweet'] = train.text.apply(normalizer)

    file_test = readfile('../../datasets/public_development_en/dev_en.tsv')
    test = pd.DataFrame()
    test['id'] = file_test[0]
    test['text'] = file_test[1]
    test['normalized_tweet'] = test.text.apply(normalizer)

    train_text = train['normalized_tweet']
    test_text = test['normalized_tweet']
    all_text = np.concatenate([train_text, test_text])
    class_names = ['hate_decision']
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 3),
        max_features=10000)
    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(train.text)
    test_word_features = word_vectorizer.transform(test.text)

    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        ngram_range=(2, 6),
        max_features=50000)
    char_vectorizer.fit(all_text)
    train_char_features = char_vectorizer.transform(train_text)
    test_char_features = char_vectorizer.transform(test_text)

    train_features = hstack([train_char_features, train_word_features])
    test_features = hstack([test_char_features, test_word_features])
    scores = []
    submission = pd.DataFrame.from_dict({'id': test['id']})

    for class_name in class_names:
        train_target = train[class_name]
        print('1')
        classifier = SVC(gamma=0.01, C=0.01, probability=True, class_weight='balanced', kernel='linear')
        print('2')
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))

        classifier.fit(train_features, train_target)
        proba_pred = classifier.predict_proba(test_features)[:, 1]
        print(len(proba_pred))
        sub_pred = []
        for i in proba_pred:
            if i >= 0.5:
                sub_pred.append('1')
            else:
                sub_pred.append('0')
        submission[class_name] = sub_pred




    '''for class_name in class_names:
        train_target = train[class_name]
        classifier = LogisticRegression(C=0.1, solver='sag')
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))

        classifier.fit(train_features, train_target)
        proba_pred = classifier.predict_proba(test_features)[:, 1]
        print(len(proba_pred))
        sub_pred = []
        for i in proba_pred:
            if i >= 0.5:
                sub_pred.append('1')
            else:
                sub_pred.append('0')
        submission[class_name] = sub_pred '''


    print('Total CV score is {}'.format(np.mean(scores)))

    submission.to_csv('submission.tsv', index=False, columns=None, sep='\t')



