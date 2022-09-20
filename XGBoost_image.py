# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 22:40:13 2022
must try for loading image set from local and then trainning.
Training with image set looks possible but accuracy is very bad need to check with other datasets as well
@author: raj.yadav
"""
from keras.datasets import cifar10
(X_train,y_train),(X_test,y_test)=cifar10.load_data()

# now the X train,X_test is of feature 50000,32,32,3 so we bring down the dimension by multiply 32*32*3=3072
X_train=X_train.reshape(50000,3072)
X_test=X_test.reshape(10000,3072)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

# COnverting the (50000,1) into (50000,) by flattening
y_train=y_train.flatten()
y_test=y_test.flatten()

# normalizing the value i.e scaling the value to range from 0 to 1
X_train/=255.
X_test/=255.

import xgboost # from xgboost import XGBClassifier

model=xgboost.XGBClassifier(gamma=0.5,learning_rate=0.01,max_delta_step=0.1,
                            max_depth=4,min_child_weight=0.2,n_estimators=10,
                            nthread=4,objective='multi:softmax',reg_alpha=0.5,
                            reg_lambda=0.8,scale_pos_weight=1, silent=False,
                            subsample=0.8)

from sklearn import metrics
from sklearn.model_selection import cross_val_score
cv_results=cross_val_score(model, X_train,y_train,
                           cv=2,scoring='accuracy',n_jobs=-1,verbose=1)
model.fit(X_train,y_train,verbose=True)

print(cv_results)
print(model)

y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred)
