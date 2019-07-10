#!/usr/bin/env python
# coding=UTF-8
'''
@Description: NaiveBayesian
@Author: bill
@LastEditors: Please set LastEditors
@Date: 2019-03-31 00:13:46
@LastEditTime: 2019-07-11 07:49:04
'''

import numpy as np
import pandas as pd
import math
from sklearn.datasets import load_iris,load_boston

class LinearRegression():

    def __init__(self, feature_num,lr = 0.001, batch_size=24, epochs= 10 ,iter_show_info=32):
        self._lr = lr
        self._W = None
        self._batch_size = batch_size
        self._epochs = epochs
        self._iter_show_info = iter_show_info
        print("init weight")
        self._W = np.random.normal(size=feature_num)
        print(self._W)

    def train(self, X, Y):
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
        return output
    
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
        print("X.T:")
        print(X.T)
        print("weight:")
        print(self._W)
        print("forword output:")
        print(forward_output)
        print("delta y:")
        print((forward_output - Y))
        print("grad:")
        print(grad)
        #exit(0)
        return grad
    
    def metric(self, X,Y):
        output = self.forward(X)
        def mae(preds,Y):
            return np.abs(preds - Y).mean()
        print("mae is %f"% mae(output, Y) )

    def optimize(self, X, Y):
        self._W = self._W - self._lr * self.get_gradient(X, Y)
        print(self._W)


if __name__ == "__main__":

    iris = load_boston() #load_iris()
    #print(type(iris))
    
    #print(iris)
    lr = LinearRegression(iris.data.shape[1])
    lr.train(iris.data, iris.target)
    lr.metric(iris.data, iris.target)

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(iris.data, iris.target)
    preds  = reg.predict(iris.data)
    mae = np.abs(preds -  iris.target).mean()
    print("sklearn mae is %f"%mae )