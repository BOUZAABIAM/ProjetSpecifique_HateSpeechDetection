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

from svm_classifier.en_classifier.en_a_classifier_1 import tokenize, normalizer_listing, normalizer


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
            tr.append(line[3])
            ag.append(line[4])
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
    input_training_files = [
        '../../datasets/public_development_en/train_en.tsv',
        '../../datasets/trial_en.tsv',
        '../../datasets/public_development_es/dev_es.tsv'
    ]
    # Read english (train + trial) data
    id = []
    tweets = []
    hate_speech = []
    target = []
    aggressive = []
    for file in input_training_files:
        data = readfile(file)
        id = id + data[0]
        tweets = tweets + data[1]
        hate_speech = hate_speech + data[2]
        target = target + data[3]


    # Create DataFrame with data
    t = pd.DataFrame()
    t['text'] = tweets
    t['hate_decision'] = hate_speech
    t['normalized_tweet'] = t.text.apply(normalizer_listing)

    # Split Data into two parts : train and test
    train, test = train_test_split(t, test_size=0.25, random_state=10)
    X_train = train['text'].values
    X_test = test['text'].values
    y_train = train['hate_decision']
    y_test = test['hate_decision']

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
    grid_svm = GridSearchCV(pipeline_svm,
                             param_grid={'svc__C': [0.01, 0.025, 0.05, 0.1, 1, 10, 100]},
                             cv=kfolds,
                             scoring="f1",
                             verbose=1,
                             n_jobs=-1)

    # fit the  svm model
    grid_svm.fit(X_train, y_train)

    # Show Results
    print(report_results(grid_svm.best_estimator_, X_test, y_test))
    #print(grid_svm.score(X_test, y_test))
    #print(grid_svm.best_params_)
    #print(grid_svm.best_score_)
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


    # Start prediction
    f = open("../svm_predictions/en_a.tsv", "w",encoding="utf8")
    neg_pred = 0
    #with open('../../datasets/public_development_en/dev_en.tsv', encoding="utf8") as tsvfile:
    with open('../../datasets/public_test_en/test_en.tsv', encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line_number, line in enumerate(tsvreader):
            if(line_number>0):
                f.write(str(line[0])+"\t")
                prediction = grid_svm.predict([normalizer(line[1])])
                proba = grid_svm.predict_proba([line[1]])

                f.write(str(prediction[0]))
                f.write('\n')


    f.close()
    #print(neg_pred)





