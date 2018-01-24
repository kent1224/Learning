# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:14:11 2017

@author: 14224
"""


#####引入函式庫及內建手寫數字資料庫#######
from sklearn import datasets,svm,metrics
import matplotlib.pyplot as plt

digits = datasets.load_digits()  #dict資料型別

####確認參數意義#####
plt.figure(1,figsize=(3,3))
plt.imshow(digits.images[-1],cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()

####observe data####
for key,value in digits.items():
    try:
        print(key,value.shape)
    except:
        print(key)


images_and_labels = list(zip(digits.images,digits.target))
for index,(image,label) in enumerate(images_and_labels[:4]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Training: %i' % label)

####描述檔####
print(digits['DESCR'])

n_samples = len(digits.images)
data = digits.images.reshape((n_samples,-1))

######################
####Classification####
######################

####設定svc參數
classifier = svm.SVC(gamma=0.001)
####前半部做訓練####
classifier.fit(data[:n_samples/2],digits.target[:n_samples/2])
####後半部的正確答案與預測值####
expected = digits.target[n_samples/2:]
predicted = classifier.predict(data[n_samples/2:])

####準確度用混淆矩陣(confusion matrix)來統計####
print("confusion matrix: \n%s" % metrics.confusion_matrix(expected,predicted))

####將confusion matrix圖示出來####
def plot_confusion_matrix(cm,title='confusion_matrix',cmap=plt.cm.Blues):
    import numpy as np
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(digits.target_names))
    plt.xticks(tick_marks,digits.target_names,rotation=45)
    plt.yticks(tick_marks,digits.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure()
plot_confusion_matrix(metrics.confusion_matrix(expected,predicted))

####用metrics物件計算precision,recall,f1-score來探討精確度####
print("classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected,predicted)))

####觀察影像以及預測(分類)結果的對應關係####
images_and_predictions = list(zip(digits.images[n_samples/2:],predicted))

for index,(image,prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('prediction: %i' % prediction)
plt.show()

###################
####特徵選擇RFE####
##################
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

X = digits.images.reshape((len(digits.images),-1))
y = digits.target

####create the RFE object and rank each pixel####
svc = SVC(kernel='linear',C=1)
rfe = RFE(estimator=svc,n_features_to_select=1,step=1)
rfe.fit(X,y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

#####把權重順序畫出來####
plt.matshow(ranking,cmap=plt.cm.Blues)
plt.colorbar()
plt.title('ranking of pixels with RFE')
plt.show()

#####################
####特徵選擇RFECV####
####################

from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

X,y = make_classification(n_samples=1000,n_features=25,n_informative=3,
                          n_redundant=2,n_repeated=0,n_classes=8,
                          n_clusters_per_class=1,random_state=0)

svc=SVC(kernel='linear')
rfecv=RFECV(estimator=svc,step=1,cv=StratifiedKFold(y,2),scoring='accuracy')
rfecv.fit(X,y)
print('optimal number of features: %d' % rfecv.n_features_)

plt.figure()
plt.xlabel('number of feature selected')
plt.ylabel('cross validation score(nb of correct classification)')
plt.plot(range(1,len(rfecv.grid_scores_)+1),rfecv.grid_scores_)
plt.show()
