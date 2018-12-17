import math
import re
import csv
from itertools import zip_longest
from datetime import datetime
from collections import Counter


def tokenize(input_file, encoding):
    lst =[]
    with open(input_file, 'r', encoding=encoding) as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for sent in tsvreader:
            sen = sent[1].lower()
            sen = re.sub("[A-z0-9\'\"`\|\/\+\#\,\)\(\?\!\B\-\:\=\;\.\Â«\Â»\--\@]", '', sen)
            sen = re.findall('\w+', sen)
            for word in sen:
                lst.append(word)
    print(lst)
    return lst


def ngrams_split(lst, n):
    return [' '.join(lst[i:i+n]) for i in range(len(lst)-n)]


def two_gram_count(tokens, n_filter, n):
    ngram_count = []
    n_words = len(tokens)
    for ngram, count in Counter(ngrams_split(tokens, n)).items():
        if count >= n_filter:
            splitted_ngram = ngram.split()
            ngram_freq = math.log(count/n_words)
            num = count*n_words
            f1 = tokens.count(splitted_ngram[0])
            f2 = tokens.count(splitted_ngram[1])
            mi = math.pow(math.log(num/(f1*f2), 10), 2)
            ngram_prob = math.log(count/f1, 10)
            ngram_count.append((ngram_freq, mi, ngram_prob, count, ngram))
    return ngram_count


def three_gram_count(tokens, n_filter, n):
    ngram_count = []
    n_words = len(tokens)
    ng = ngrams_split(tokens, 2)
    for ngram, count in Counter(ngrams_split(tokens, n)).items():
        if count >= n_filter:
            splitted_ngram = ngram.split()
            ngram_freq = math.log(count/n_words, 10)
            num = count*n_words
            c2gram = ng.count(splitted_ngram[0] + " " + splitted_ngram[1])
            f1 = tokens.count(splitted_ngram[0])
            f2 = tokens.count(splitted_ngram[1])
            f3 = tokens.count(splitted_ngram[2])
            mi = math.pow(math.log(num/(f1*f2*f3), 10), 2)
            ngram_prob = math.log(count/c2gram, 10)
            ngram_count.append((ngram_freq, mi, ngram_prob, count, ngram))
    return ngram_count


def four_grams_count(tokens, n_filter, n):
    ngram_count = []
    n_words = len(tokens)
    ng2 = ngrams_split(tokens, 2)
    for ngram, count in Counter(ngrams_split(tokens, n)).items():
        if count >= n_filter:
            splitted_ngram = ngram.split()
            ngram_freq = math.log(count/n_words, 10)
            num = count*n_words
            c1gram = ng2.count(splitted_ngram[0] + " " + splitted_ngram[1])
            c2gram = ng2.count(splitted_ngram[1] + " " + splitted_ngram[2])
            c3gram = ng2.count(splitted_ngram[2] + " " + splitted_ngram[3])
            f1 = tokens.count(splitted_ngram[0])
            f2 = tokens.count(splitted_ngram[1])
            f3 = tokens.count(splitted_ngram[2])
            f4 = tokens.count(splitted_ngram[3])
            mi = math.pow(math.log(num/(f1*f2*f3*f4), 10), 2)
            prob1 = c1gram/f1
            prob2 = c2gram/f2
            prob3 = c3gram/f3
            ngram_prob = math.log(prob1, 10) + math.log(prob2, 10) +    math.log(prob3, 10)
            ngram_count.append((ngram_freq, mi, ngram_prob, count, ngram))
    return ngram_count


def n_grams_stat(input_file, encoding, n_filter, n):
    tokens = tokenize(input_file, encoding)
    if n == 2:
        return two_gram_count(tokens, n_filter, n)
    elif n == 3:
        return three_gram_count(tokens, n_filter, n)
    elif n == 4:
        return four_grams_count(tokens, n_filter, n)
    return []


if __name__ == "__main__":
    s = n_grams_stat("../datasets/trial_en.tsv",'utf8', n_filter=3, n=2)
    for a, b, c, d, e in s:
        print(a, b, c, d, e)
