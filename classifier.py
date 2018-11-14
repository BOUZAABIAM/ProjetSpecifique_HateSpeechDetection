import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


import pandas as pd
import csv
import re, nltk
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ngrams
import collections
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from nltk.tokenize import TweetTokenizer
pd.set_option('display.max_colwidth', -1)
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score, accuracy_score


def readfile(path):
    tweets = []
    hs = []
    tr = []
    ag = []
    with open(path, encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            tweets.append(line[1])
            hs.append(line[2])
            tr.append(line[3])
            ag.append(line[4])
        tweets.pop(0)
        hs.pop(0)
        tr.pop(0)
        ag.pop(0)
    return tweets, hs, tr, ag

def tokenize(text):
    tknzr = TweetTokenizer()
    print(tknzr.tokenize(text))
    return tknzr.tokenize(text)


def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def normalizer(tweet):
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
    only_letters = clean_tweet(tweet)
    tokens = nltk.word_tokenize(only_letters)
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    #print(filtered_result)
    lemmas = ' '.join(wordnet_lemmatizer.lemmatize(t) for t in filtered_result)
    #print(lemmas)
    return lemmas


def normalizer_listing(tweet):
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
    only_letters = clean_tweet(tweet)
    tokens = nltk.word_tokenize(only_letters)
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    #print(filtered_result)
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    #print(lemmas)
    return lemmas


def clean_my_tweet(tweet):
    only_letters = clean_tweet(tweet)
    tokens = nltk.word_tokenize(only_letters)
    lower_case = [l.lower() for l in tokens]
    cleaned = ' '.join(t for t in lower_case)
    return cleaned



def ngrams(input_list):
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams + trigrams


def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt


def marginal(p):
    top2 = p.argsort()[::-1]
    return abs(p[top2[0]]-p[top2[1]])

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


def sentiment2target(sentiment):
    return {
        '0': 0,
        '1': 1,
    }[sentiment]


if __name__ == "__main__":
    tweets, hate_speech, target, aggressive = readfile('./datasets/trial_en.tsv')
    t = pd.DataFrame()
    t['text'] = tweets
    t['hate_decision'] = hate_speech
    t['target'] = target
    t['aggressive'] = aggressive
    stop_words = set(stopwords.words('english'))
    t['cleaned_tweet']=t.text.apply(clean_my_tweet)
    t['normalized_tweet'] = t.text.apply(normalizer)
    #print(t['normalized_tweet'])
    t['grams'] = t.normalized_tweet.apply(ngrams)
    #print(t['grams'])
    count_vectorizer = CountVectorizer(ngram_range=(1, 3))

    # f = open("./gen_features/features.tsv", "w")
    # for ngram in t[(t.hate_decision == '1')][['grams']].apply(count_words)['grams'].most_common():
    #     f.write(str(ngram))
    #     f.write('\n')
    # t[(t.hate_decision == '0')][['grams']].apply(count_words)['grams'].most_common(50)

    print("=============================== Processing SVM without sentiment analysis ===============================")

    '''
        Transform each sentence into a vector.The vector is of the same length as our
        vocabulary. If a particular word is present, that entry in the vector is 1, otherwise 0
    '''

    x = PrettyTable()
    vectorized_data = count_vectorizer.fit_transform(t.cleaned_tweet)
    features = count_vectorizer.get_feature_names()
    x.field_names = features
    for row in vectorized_data.toarray():
        x.add_row(row)
    table_txt = x.get_string()
    with open('./gen_features/Tweet_ngram1.tsv', 'w') as file:
        file.write(table_txt)

    indexed_data = hstack((np.array(range(0, vectorized_data.shape[0]))[:, None], vectorized_data))
    targets = t.hate_decision.apply(sentiment2target)
    data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.4, random_state=10)
    data_train_index = data_train[:, 0]
    data_train = data_train[:, 1:]
    data_test_index = data_test[:, 0]
    data_test = data_test[:, 1:]
    clf = OneVsOneClassifier(svm.SVC(gamma=0.01, C=100, probability=True, class_weight='balanced', kernel='linear'))
    clf_output = clf.fit(data_train, targets_train)
    '''
        The mean accuracy on the given test data
    '''
    print('The mean accuracy on the given test data : '+str(clf.score(data_test, targets_test)))

    print("=============================== Processing SVM using sentiment analysis ===============================")
    x = PrettyTable()
    vectorized_data = count_vectorizer.fit_transform(t.normalized_tweet)
    features = count_vectorizer.get_feature_names()
    x.field_names = features
    for row in vectorized_data.toarray():
        x.add_row(row)
    table_txt = x.get_string()
    with open('./gen_features/Tweet_ngram_SA.tsv', 'w') as file:
        file.write(table_txt)

    indexed_data = hstack((np.array(range(0, vectorized_data.shape[0]))[:, None], vectorized_data))
    targets = t.hate_decision.apply(sentiment2target)
    data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.4,
                                                                          random_state=10)
    data_train_index = data_train[:, 0]
    data_train = data_train[:, 1:]
    data_test_index = data_test[:, 0]
    data_test = data_test[:, 1:]
    clf = OneVsOneClassifier(svm.SVC(gamma=0.01, C=100, probability=True, class_weight='balanced', kernel='linear'))
    clf_output = clf.fit(data_train, targets_train)
    print('The mean accuracy on the given test data : '+str(clf.score(data_test, targets_test)))



