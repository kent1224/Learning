# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:37:21 2017

@author: 14224
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

def evaluate_and_get_testing_label(trainx,testx,trainy,testy,testxx):
    
    normalizer = Normalizer().fit(trainx)
    trainx = normalizer.transform(trainx)
    testx = normalizer.transform(testx)
    testxx = normalizer.transform(testxx)
    
    pca = PCA(n_components=0.8, whiten=True).fit(trainx)
    trainx = pca.transform(trainx)
    testx = pca.transform(testx)    
    testxx = pca.transform(testxx)
    
    svmclf = svm.SVC(C=2.8).fit(trainx, trainy)
    
    return svmclf.score(testx, testy), svmclf.predict(testxx)

def write_to_submission(testing_label):
    testing_label_df = pd.DataFrame(testing_label, columns=['Label'])
    testing_label_df.index +=1
    testing_label_df.index.name = 'ImageId'
    testing_label_df.to_csv('sample_submission.csv', sep=',')

def main():
    
    df = pd.read_csv("C:/Users/14224/Desktop/Kaggle_digit recognizer/train.csv")
    df_test = pd.read_csv("C:/Users/14224/Desktop/Kaggle_digit recognizer/test.csv")
    train_label = df['label'][:1500]
    train_feature = df.drop('label',axis=1)[:1500]
    testxx = df_test 
    
    
    trainx, testx, trainy, testy = train_test_split(
         train_feature, train_label, test_size = 0.2, random_state = 0)

    
    score, aa = evaluate_and_get_testing_label(trainx,testx,trainy,testy,testxx)
    
    print("score = {}".format(score))
    
    write_to_submission(aa)
         
if __name__ == "__main__":
    main()


