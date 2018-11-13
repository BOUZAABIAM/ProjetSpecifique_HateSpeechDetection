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
from sklearn.multiclass import OneVsOneClassifier

pd.set_option('display.max_colwidth', -1)


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


def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def normalizer(tweet):
    wordnet_lemmatizer = WordNetLemmatizer()
    only_letters = clean_tweet(tweet)
    tokens = nltk.word_tokenize(only_letters)
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas


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
    t['normalized_tweet'] = t.text.apply(normalizer)
    # print(t)
    t['grams'] = t.normalized_tweet.apply(ngrams)
    # print(t['grams'])

    f = open("./gen_features/features.tsv", "w")
    for ngram in t[(t.hate_decision == '1')][['grams']].apply(count_words)['grams'].most_common():
        f.write(str(ngram))
        f.write('\n')
    t[(t.hate_decision == '0')][['grams']].apply(count_words)['grams'].most_common(50)

    '''
    Transform each sentence into a vector.The vector is of the same length as our
    vocabulary. If a particular word is present, that entry in the vector is 1, otherwise 0
    '''
    count_vectorizer = CountVectorizer(ngram_range=(1, 3))
    vectorized_data = count_vectorizer.fit_transform(t.text)
    '''print("===============================")
    print(count_vectorizer.get_feature_names())
    print("===============================")
    print(vectorized_data.toarray())
'''
    indexed_data = hstack((np.array(range(0, vectorized_data.shape[0]))[:, None], vectorized_data))
    targets = t.hate_decision.apply(sentiment2target)
    data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.5,
                                                                          random_state=0)
    data_train_index = data_train[:, 0]
    data_train = data_train[:, 1:]
    data_test_index = data_test[:, 0]
    data_test = data_test[:, 1:]

    clf = OneVsOneClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
    clf_output = clf.fit(data_train, targets_train)
    print(clf.score(data_test, targets_test))
