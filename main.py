import matplotlib.pyplot as plt
import csv
from svm_classifier.preprocessing import *


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


def stats(list,labels):
    slices = [list.count('1'), len(list) - list.count('1')]
    print(slices)
    colors = ['r', 'b']
    # plotting the pie chart
    plt.pie(slices, labels=labels, colors=colors, startangle=90, shadow=True, explode=(0, 0), radius=1.2,
            autopct='%1.1f%%')
    # plotting legend
    plt.legend()
    # showing the plot
    plt.show()


ENG_FILE = './datasets/trial_en.tsv'


def main():
    tweets, hate_speech, target, aggressive = readfile(ENG_FILE)

    # Stats ====================================================================================================
    stats(hate_speech, ['Hate', 'Non Hate'])
    stats(target, ['Individual', 'Group'])
    stats(aggressive, ['Aggressive', 'Non Aggressive'])
    print(count_twitter_objs(tweets[0]))


main()
