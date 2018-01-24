# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 08:22:59 2017

@author: 14224
"""

""" ML x sklearn - template def """

import pandas as pd
import numpy as np
from sklearn import metrics, linear_model
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

def data_exploration(xtrain, xtest, ytrain, ytest, x_prediction):
    
    
    # feature selection
    pca = PCA(n_components = 5, whiten = True).fit(xtrain)
    xtrain = pca.transform(xtrain)
    xtest = pca.transform(xtest)
    x_prediction = pca.transform(x_prediction)
    
    # feature transformation
    normalizer = Normalizer().fit(xtrain)
    xtrain = normalizer.transform(xtrain)
    xtest = normalizer.transform(xtest)
    x_prediction = normalizer.transform(x_prediction)
    
    return xtrain, xtest, x_prediction

def performance_function():

    
def train_predict_score(xtrain,xtest,ytrain,ytest,x_prediction):
    
    """ fit data to the model to train """
    print("Training...")
    
    model = SVC(probability = True).fit(xtrain, ytrain)
    
    """ predict """
    print("Predicting...")
    
    #y_prediction = model.predict_proba(x_prediction)
    y_prediction = model.decision_function(x_prediction)    
    
    """ score """
    score = model.score(xtest,ytest)    
    
    """ print results """
    
    print("Score is {}".format(score))
    print(y_prediction)
    

def main():
    
    """ Load data """
    print("Loading data...")
    
    # Load data
    training_data = pd.read_csv('C:/Users/14224/Desktop/Numerai/Tournament 51/numerai_datasets/numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('C:/Users/14224/Desktop/Numerai/Tournament 51/numerai_datasets/numerai_tournament_data.csv', header=0)

    # split data into x, y, train, validation, test, ...
    features = [f for f in list(training_data) if "feature" in f]
    X = training_data[features][:1000]
    Y = training_data["target"][:1000]
    x_prediction = prediction_data[features]
    
    xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size = 0.2, random_state =0)
    
    """ data exploration """
    xtrain, xtest, x_prediction = data_exploration(xtrain, xtest, x_prediction)
    
    """ define evaluating performance function """
    
    """ train, predict, and score """
    train_predict_score(xtrain,xtest,ytrain,ytest,x_prediction)


if __name__ == '__main__':
    main()
    