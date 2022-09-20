# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:28:14 2022

@author: raj.yadav
"""

import pandas as pd
data=pd.read_csv('Xgboost_dataset.csv')
data=data.drop('User ID',axis=1)
data.head()
feature=data.iloc[:,1:3]
target=data.iloc[:,3]       #or can be wriiten as #data.iloc[:,3:]

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train,X_test,y_train,y_test=train_test_split(feature,target,shuffle=True,random_state=7,test_size=0.3)
model=XGBClassifier()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)
score=accuracy_score(y_test, y_pred)
print(y_test) # y_test[0] throwing error
data1={'Age': [28],
       'Salary':[10000]}
data1=pd.DataFrame(data1)
y_pred1=model.predict(data1)

  





















