#!/usr/bin/env python
# coding=UTF-8
'''
@Description: NaiveBayesian
@Author: bill
@LastEditors: Please set LastEditors
@Date: 2019-03-31 00:13:46
@LastEditTime: 2019-07-14 23:10:14
'''

import numpy as np
import pandas as pd
import math
from sklearn.datasets import load_iris,load_boston,load_breast_cancer
from LinearRegression import LinearRegression

class LogisticRegression(LinearRegression):

    def __init__(self, feature_num,lr = 0.001, batch_size=1, epochs= 500 ,iter_show_info=320):
        self._lr = lr
        self._W = None
        self._batch_size = batch_size
        self._epochs = epochs
        self._iter_show_info = iter_show_info
        print("init weight")
        self._W = np.random.normal(size=(feature_num + 1) )
        print(self._W)

    def _sigmoid(self, X):
        #print(X)
        return 1.0 / (1+ np.exp(-X) )

    def train(self, X, Y):
        X = np.column_stack([X, np.ones(X.shape[0])])
        for epoch_num in range(self._epochs):
            #batch_ind = 0 
            batch_num = math.ceil(X.shape[0]/self._batch_size)
            #for batch_x, batch_y in zip(X[::self._batch_size], Y[::self._batch_size]):
            for batch_ind in range(batch_num):

                batch_x = X[batch_ind:batch_ind+self._batch_size]
                batch_y = Y[batch_ind:batch_ind+self._batch_size]
                #print(batch_x)
                self.optimize(batch_x, batch_y)
                forward_output = self.forward(batch_x)
                loss = self.get_loss(forward_output, batch_y)
                if (epoch_num * batch_num + batch_ind ) % self._iter_show_info == 0:
                    print("epoch %d batch_index %d loss is %f"%(epoch_num, batch_ind, loss))

    
    def forward(self, X):
        output = np.dot(X, self._W.T ) 
        return self._sigmoid(output)
    
    def backward(self, X, Y):
        self.optimize(X,Y)

    def get_loss(self, output, Y):
        loss = np.dot((Y - output).T,(Y - output)).mean()
        if np.isnan(loss):
            exit(0)
        return np.abs(loss)
    
    def get_gradient(self, X, Y):
        forward_output = self.forward( X)
        grad = (X.T*(Y - forward_output)).mean(axis=1)
        return grad
    
    def metric(self, X,Y,threhold=0.5):
        X = np.column_stack([X, np.ones(X.shape[0])])
        output = self.forward(X)
        output[output >=threhold] = 1
        output[output < threhold] = 0
        def acc(preds,Y):
            return list(preds==Y).count(True)*1.0/len(Y)
        print("acc is %f"% acc(output, Y) )

    def optimize(self, X, Y):
        grad =  self.get_gradient(X, Y)
        self._W = self._W + self._lr * grad
        #print(self._W)
def feature_normalize(X):
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    X_norm = (X - mean) / std
    return X_norm

if __name__ == "__main__":

    iris = load_breast_cancer() #load_boston() #load_iris()
    #print(type(iris))
    #print(iris.target)
    data = feature_normalize(iris.data)
    lr = LogisticRegression(iris.data.shape[1])
    lr.train(data, iris.target)
    lr.metric(data, iris.target)

    # from sklearn.linear_model import LinearRegression
    # reg = LinearRegression().fit(iris.data, iris.target)
    # preds  = reg.predict(iris.data)
    # mae = np.abs(preds -  iris.target).mean()
    # print("sklearn mae is %f"%mae )