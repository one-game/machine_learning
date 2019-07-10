#!/usr/bin/env python
# coding=UTF-8
'''
@Description: NaiveBayesian
@Author: bill
@LastEditors: Please set LastEditors
@Date: 2019-03-31 00:13:46
@LastEditTime: 2019-03-31 02:41:53
'''
from collections import defaultdict
import math
import numpy as np

class XXNaiveBayesian:

    def __init__(self):

        self.vocabs = defaultdict(int)
        self.prob_word_con_labels = defaultdict(dict)
        self.label_prior = defaultdict(int)
        pass
    
    def train(self, train_X, train_y):
        counter = 0
        for sample, y in zip(train_X, train_y):
            counter += 1
            if counter % 1000 == 0:
                print("finish training %d lines"%(counter))
            for word in sample:
                #
                self.vocabs[word] += 1
                self.label_prior[y] += 1
                if word not in self.prob_word_con_labels[y]:
                    self.prob_word_con_labels[y][word] = 0
                self.prob_word_con_labels[y][word] += 1

                #normalize prob


    def predict(self,test_X):

        test_Y = []
        for sample in test_X:
            l_probs = []
            for label, l_prior in self.label_prior.items():
                l_prob = 0.0
                for word in sample:
                    if word not in self.vocabs or word not in self.prob_word_con_labels[label]:
                        l_prob += (-math.log(1.0/len(self.vocabs)))
                    else:
                        l_prob += (-math.log(self.prob_word_con_labels[label][word]))
                l_probs.append((label, l_prob))
            test_Y.append(l_probs)
        return test_Y

import pandas as pd
import jieba
data_df = pd.read_csv("waimai_10k.csv")
print(data_df.shape[0])
train_df = data_df[:int(data_df.shape[0]*.8)]
test_df = data_df[int(data_df.shape[0]*.8):]
stopwords = set()
with open("stopwords.txt","r") as fd:
    for line in fd:
        stopwords.add(line.strip("\n"))

if __name__ == "__main__":
    # train_X = [
    #     [1,2,3,4,5],
    #     [2,3,1,23,4],
    #     [2,123,12,4,5],
    #     [2,123,31,4,1],
    # ]
    # train_Y = [1,1,2,2]

    # test_X = [
    #     [3,123,1,32,1]
    # ]
    data_X = [filter(lambda w: w not in stopwords, jieba.lcut(sample)) for sample in data_df.review.values]
    


    # nb = XXNaiveBayesian()
    # nb.train(train_X,train_Y)
    # pred_ys = nb.predict(test_X)
    # correct_count = 0
    # for y, pred_y  in zip(test_Y, pred_ys):
    #     pred_dict = dict(pred_y)
    #     if pred_dict[y] > pred_dict[(1-y)**2]:
    #         correct_count += 1
    # print("precision is %d/%d"%(correct_count, test_df.shape[0]))
    vocab = set([word for sample in data_X for word in sample])
    vocab_dict = dict(zip(vocab, map(str,list(range(len(vocab))))))
    data_X = [map(vocab_dict.get, filter(lambda w: w not in stopwords, jieba.lcut(sample))) for sample in data_df.review.values]
    
    train_X = [map(vocab_dict.get, filter(lambda w: w not in stopwords, jieba.lcut(sample))) for sample in train_df.review.values]
    train_Y = train_df.label.values
    test_X = [map(vocab_dict.get, filter(lambda w: w not in stopwords, jieba.lcut(sample))) for sample in test_df.review.values]
    test_Y = test_df.label.values
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    feature_label = TfidfVectorizer()
    feature_label.fit(data_X)
    train_X = feature_label.transform(train_X)
    test_X = feature_label.transform(test_X)
    print(train_X.shape)
    print(train_X[1])
    model = MultinomialNB()
    model.fit(np.array(train_X), np.array(train_Y))
    print(model.predict(test_X))


