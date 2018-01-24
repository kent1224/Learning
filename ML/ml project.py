# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:53:45 2017

@author: 14224
"""
 
import sys
import pymssql
import pyodbc
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from sklearn.feature_selection import  SelectKBest, chi2, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import ShuffleSplit

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

############################################################################################################################################################################################################
""" Load Data """
print("Loading data...")

#conn = pyodbc.connect('Driver={SQL Server};Server=10.216.42.1;Database=ML;Trusted_Connection=yes;')
connstr = pymssql.connect(host = r'10.216.42.1')
#connstr = pymssql.connect(server='10.216.42.1', user='warrant', password='warrant', database='WARRANT', timeout=0, charset='UTF-8')
cursor = connstr.cursor()
#sql= "select top 1000 [券商] from [ML].[dbo].[DailyAttackedTable]"
# [券商]
sql = "SELECT [日期],[代號],[被攻擊清單],[標的代號],[標的收盤價],[權證收盤價]"
sql = sql + ",[上市日期],[到期日期],[最新履約價],[最新執行比例]"
sql = sql + ",[近三月歷史波動率(%)],[發行波動率(%)],[Bid-IV],[Ask-IV],[IV]"
sql = sql + ",[3M權證理論價],[IV權證理論價],[3MDelta值],[IVDelta值],[3MGamma值],[IVGamma值]"
sql = sql + ",[3MVega值],[IVVega值],[3MTheta值],[IVTheta值],[流通數量(千)]"
sql = sql + ",[距到期日天數],[3M毛利率],[權證跳動價差],[交易稅],[理論價差],[Spread]"
sql = sql + ",[現股tick反應比例(%)],[全部成本反應比例(%)],[現股價差百分位數],[全部價差百分位數]"
sql = sql + ",[現股買1平均單量],[現股賣1平均單量],[佈單Delta金額],[單檔佈單Delta金額百分位數] "
sql = sql + ",[現股佈單金額(千)],[券商標的佈單總金額] "
sql = sql + "from [ML].[dbo].[DailyAttackedTable] "
sql = sql + "where [日期] >= '20170202'"


#sql = sql.encode('utf-8').strip()
#df = pd.io.sql.read_sql(sql,conn)
#conn.close()
df = pd.read_sql(sql,connstr)
cursor.close()
connstr.close()

df = df.sort(['代號','日期']).reset_index().drop('index',axis=1)
df.loc[:,'被攻擊清單1'] = pd.DataFrame(df['被攻擊清單'].ix[1:len(df['被攻擊清單'])-1]).set_index(np.arange(0,len(df['被攻擊清單'])-1)).rename(columns={'被攻擊清單':'被攻擊清單1'}).fillna(0)
df.loc[:,'代號1'] = pd.DataFrame(df['代號'].ix[1:len(df['代號'])-1]).set_index(np.arange(0,len(df['代號'])-1)).rename(columns={'代號':'代號1'}).fillna(0)
df = df[df['代號1']==df['代號']].reset_index().drop(['被攻擊清單','代號1'],axis=1).rename(columns={'被攻擊清單1':'被攻擊清單'}).drop('index',axis=1)

# split data into x, y
#y = df['label'][:1000] 
y = df['被攻擊清單'].reset_index().drop('index',axis=1)
x = df.drop(['被攻擊清單','代號'],axis=1)

###########################################################################################################################################################################################################
""" understand the data """
print(x.shape)
print(x.describe())

# class distribution
print(y[y['被攻擊清單']=='0'].count())
print(y[y['被攻擊清單']=='1'].count())

print(x.corr(method='pearson'))
print(x.skew())

#visualization
#univariate plots
x.hist()
x.plot(kind = 'density', subplots = True, layout = (6,6), sharex = False)
x.plot(kind = 'box', subplots = True, layout = (6,6), sharex = False, sharey = False)
plt.show()

#multivariate plots
#correlation matrix plot
names = []
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(x.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
#ticks = np.arange(0,36,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
plt.show()

#scatter plot matrix
pd.tools.plotting.scatter_matrix(x)

############################################################################################################################################################################################################
""" preprocess the data """
# rescale:
#This is useful for optimization algorithms used in the core of machine learning algorithms like gradient
#descent. It is also useful for algorithms that weight inputs like regression and neural networks
#and algorithms that use distance measures like k-Nearest Neighbors.
x = MinMaxScaler(feature_range = (0,1)).fit_transform(x)
print(x)

#standardize:
#It is most suitable for techniques that assume a Gaussian
#distribution in the input variables and work better with rescaled data, such as linear regression,
#logistic regression and linear discriminate analysis.
x = StandardScaler.fit_transform(x)
print(x)

#normalize: rescaling each observation (row) to have a length of 1 (called a unit norm or a vector with the length of 1 in linear algebra).
#This pre-processing methodcan be useful for sparse datasets (lots of zeros) with attributes of varying scales when using
#algorithms that weight input values such as neural networks and algorithms that use distance
#measures such as k-Nearest Neighbors.
x = Normalizer().fit_transform(x)
print(x)

#binarize: All values above the threshold are marked 1 and all equal to or below are marked as 0.
#It can be useful when you have probabilities that you want to make crisp
#values. It is also useful when feature engineering and you want to add new features that indicate
#something meaningful.
x = Binarizer(threshold=0.0).fit_transform(x)
print(x)

# one-hot encode


# generate polynomial features
#poly = PolynomialFeatures(degree=2)
#xtrain = poly.fit_transform(xtrain)
#xtest = poly.fit_transform(xtest)

#df1 = pd.DataFrame(xtrain)
############################################################################################################################################################################################################
""" feature selection """
#Having irrelevant features in your data can decrease the accuracy of many models, 
#especially linear algorithms like linear and logistic regression.

#univariate selection: Statistical tests can be used to select those features that have the strongest relationship with the output variable.
kbest = SelectKBest(score_func = chi2, k=4).fit(x,y) #uses the chi-squared (chi2) statistical test for non-negative features to select 4 of the best features
print(kbest.scores_)
x = kbest.transform(x)

# Recursive feature elimination(RFE): recursively removing attributes and building a model on those attributes that remain. It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.
model = LogisticRegression()
rfe = RFE(model,3).fit(x,y) # uses RFE with the logistic regression algorithm to select the top 3 features. The choice of algorithm does not matter too much as long as it is skillful and consistent.
print("number feature: %d" %rfe.n_features_)
print("Selected Features: %s" % rfe.support_)
print("Feature Ranking: %s" % rfe.ranking_)

# principal component analysis(PCA):uses linear algebra to transform the dataset into a compressed form. Generally this is called a data reduction technique.
pca = PCA(n_components=10, whiten=True).fit(x) # use PCA and select 3 principal components.
x = pca.transform(x)

print("Explained Variance: %s") % pca.explained_variance_ratio_
print(pca.components_)

df2 = pd.DataFrame(x)

# feature importance: Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.
etc = ExtraTreesClassifier().fit(x,y)
print(etc.feature_importances_)
############################################################################################################################################################################################################
""" model selection """
# split data into train and test 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 7)

# kfold cross validation
kfold = KFold(len(x), n_folds = 10, random_state = 7)
#model = LogisticRegression()
#results = cross_val_score(model, x, y, cv = kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

# leave-one-out cross validation
loo = LeaveOneOut()
#model = LogisticRegression()
#results = cross_val_score(model, x, y, cv = loo)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

# repeated random test-train splits
kfold = ShuffleSplit(n_splits = 10, test_size = 0.33, random_state = 7)
#model = LogisticRegression()
#results = cross_val_score(model, x, y, cv = kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

# 1. Generally k-fold cross-validation is the gold standard for evaluating the performance of a machine learning algorithm on unseen data with k set to 3, 5, or 10.
# 2. Using a train/test split is good for speed when using a slow algorithm and produces performance estimates with lower bias when using large datasets.
# 3. Techniques like leave-one-out cross-validation and repeated random splits can be useful intermediates when trying to balance variance in the estimated performance, model training speed and dataset size.
#The best advice is to experiment and 
#nd a technique for your problem that is fast and produces reasonable estimates of performance that you can use to make decisions. If in doubt, use 10-fold cross-validation.

################################################################################################################################################################################################################
""" define evaluating performance function """
""" classification matrix """
# classification accuracy: 
#It is really only suitable when there are an equal number of observations in each class
#(which is rarely the case) and that all predictions and prediction errors are equally important,
#which is often not the case.

#kfold = KFold(n_splits = 10, random_state = 7)
#model = LogisticRegression()
results = cross_val_score(model, x, y, cv = kfold, scoring = 'accuracy')
print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())

# logarithmic loss: a performance metric for evaluating the predictions of probabilities of membership to a given class.
# The scalar probability between 0 and 1 can be seen as a measure of con
#dence for a prediction by an algorithm.
results = cross_val_score(model, x, y, cv = kfold, scoring = 'neg_log_loss')
print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())

""" regression matrix """

#################################################################################################################################################################################################################
""" fit data to the model to train """
#print("Training...")
#lgclf = LogisticRegression(solver = 'newton-cg',C=1).fit(xtrain, ytrain)

models = []
# linear classification algorithms:
models.append(('LR', LogisticRegression())) 
# assumes a Gaussian distribution for the numeric input variables and can model binary classication problems.
models.append(('LDA', LinearDiscriminantAnalysis())) 
# statistical technique for binary and multiclass classication. It too assumes a Gaussian distribution for the numerical input variables.

# nonlinear classification algorithms:
models.append(('KNN', KNeighborsClassifier())) 
# a distance metric to nd the k most similar instances in the training data for a new instance and takes the mean outcome of the neighbors as the prediction.
#models.append(('CART', DecisionTreeClassifier())) # construct a binary tree from the training data.
#models.append(('NB', GaussianNB())) # calculates the probability of each class and the conditional probability of each class given each input value.
models.append(('SVM', SVC())) 
# seek a line that best separates two classes.

# linear regression algorithms:

# nonlinear regression algorithms:


scores = []
names = []
times = []

for name, model in models:
    print("Training...", name)
    tstart = time.time()
    #fit_model = model.fit(xtrain, ytrain)
    score = cross_val_score(model, x, y, cv = kfold, scoring = 'accuracy')
    
    tend = time.time()
    #score = fit_model.score(xtest, ytest)
    scores.append(score)
    names.append(name)
    times.append(tend-tstart)

""" predict """
#print("Predicting...")
#aa = lgclf.predict(xxtest)

""" score """
#score = lgclf.score(xtest, ytest)

""" print results """
for i in range(len(names)):
    print (names[i], ": ",scores[i].mean(), "(",scores[i].std(), ") ", "cost ",times[i], "secs")
#print('score = {}'.format(score))
#print(aa)

""" plot the results """
fig = plt.figure()
fig.suptitle('algorithms comparison')

ax = fig.add_subplot(111)
plt.boxplot(scores)
#ax.plot(scores)
ax.set_xticklabels(names)
plt.show()