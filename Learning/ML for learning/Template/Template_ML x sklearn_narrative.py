# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:25:46 2017

@author: 14224
"""

""" ML x sklearn - template narrative"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import train_test_split


""" Load Data """
print("Loading data...")

# load data
df = pd.read_csv("C:/Users/14224/Desktop/Kaggle_digit recognizer/train.csv")
df_test = pd.read_csv("C:/Users/14224/Desktop/Kaggle_digit recognizer/test.csv")

# split data into x, y, train, validation, test, ... 
y = df['label'][:1000] 
x = df.drop('label',axis=1)[:1000]
xxtest = df_test[:1000]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)


""" Data Exploration """
print("Explorating data")
# feature selection
pca = PCA(n_components=50, whiten=True).fit(xtrain)
xtrain = pca.transform(xtrain)
xtest = pca.transform(xtest)
xxtest = pca.transform(xxtest)    

# data transformation
poly = PolynomialFeatures(degree)
xtrain = poly.fit_transform(xtrain)
xtest = poly.fit_transform(xtest)
xxtest = poly.fit_transform(xxtest)

""" define evaluating performance function """


""" fit data to the model to train """
print("Training...")
lgclf = LogisticRegression(solver = 'newton-cg',C=c).fit(xtrain, ytrain)

""" predict """
print("Predicting...")
aa = lgclf.predict(xxtest)

""" score """
score = lgclf.score(xtest, ytest)

""" print results """
print('score = {}'.format(score))
print(aa)