# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:34:07 2017

@author: 14224
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import train_test_split

def do(c=1.0, degree=2):
    df = pd.read_csv("C:/Users/14224/Desktop/Kaggle_digit recognizer/train.csv")
    df_test = pd.read_csv("C:/Users/14224/Desktop/Kaggle_digit recognizer/test.csv")
    y = df['label'][:1000] 
    x = df.drop('label',axis=1)[:1000]
    xxtest = df_test[:1000]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
    
    pca = PCA(n_components=50, whiten=True).fit(xtrain)
    xtrain = pca.transform(xtrain)
    xtest = pca.transform(xtest)
    xxtest = pca.transform(xxtest)    
    
    poly = PolynomialFeatures(degree)
    xtrain = poly.fit_transform(xtrain)
    xtest = poly.fit_transform(xtest)
    xxtest = poly.fit_transform(xxtest)
    
    lgclf = LogisticRegression(solver = 'newton-cg',C=c).fit(xtrain, ytrain)
    
    score = lgclf.score(xtest, ytest)
    aa = lgclf.predict(xxtest)
    
    print('score = {}'.format(score))
    print(aa)

do()
    