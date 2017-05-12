
# coding: utf-8

# In[ ]:

##### Import Pacakges
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
from IPython.display import Image
import sklearn
from sklearn import tree
import math
#from dstools import data_tools

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 14, 10


# In[ ]:

data_dept = pd.DataFrame.from_csv('processed_data.csv')


# In[ ]:

#Set X and Y
X = data_dept.drop('Target_Variable',1)
Y = data_dept['Target_Variable']
#Create Date Splits
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, train_size =.75)


# In[ ]:

#Test Initial Naive Bayes Classifer

BN_model = BernoulliNB()
BN_model.fit(X_train,Y_train)

print(metrics.accuracy_score(Y_test,BN_model.predict(X_test)))
#0.568


# In[ ]:

# Tune BN by finding the best parameters with grid search

alpha_list =[1,2,3,4,5,6,7,8,9,10,11,12]

grid = {'alpha': alpha_list}

BN_model = BernoulliNB()
BN_model_clf = GridSearchCV(BN_model, grid)
BN_model_clf.fit(X,Y)

print(BN_model_clf.best_estimator_.alpha)
#7


# In[ ]:

#Tuned BN
BN_model = BernoulliNB(alpha=7)
BN_model.fit(X_train,Y_train)
metrics.accuracy_score(Y_test,BN_model.predict(X_test))

#Cross Val BN - roc 
print(np.mean(cross_val_score(BN_model, X, Y, cv=5)))
#0.526

