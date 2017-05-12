
# coding: utf-8

# In[ ]:

#Import Pacakges
import pandas as pd
from pandas.tseries.offsets import *
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

from math import ceil
import numpy as np
#from dstools import data_tools

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 8, 6


# In[ ]:

#Read sample data -- Department 1 ~ 20 of all stores
data = pd.read_csv('data_merge.csv')
data_dept = data[data['Dept']<=20]  


# In[ ]:

#Convert Sales Date
data_dept.Sales_Date = pd.to_datetime(data_dept.Sales_Date)


# In[ ]:

#Compute moving average of sales
#Read CSV Data

data = pd.read_csv('data_merge.csv')
data_dept = data[data['Dept']<=20]  

def moving_average(data_dept):
    avgs = {}
    for r in data_dept.index:
        store = data_dept.loc[r].Store
        dept =  data_dept.loc[r].Dept
        date = data_dept.Sales_Date.dt.date.loc[r]
        s = 0
        c = 0
        new_date = date - DateOffset(months = 1)
        temp_d = data_dept[(data_dept.Store == store )&(data_dept.Dept == dept) 
                           &(data_dept.Sales_Date.dt.year == new_date.year) 
                           & (data_dept.Sales_Date.dt.month == new_date.month)]
        s = s + sum(temp_d.Weekly_Sales)
        c = c + len(temp_d)
    

    if c != 0:
        avgs[r] = s/c
    else:
        avgs[r] = data_dept.loc[r].Weekly_Sales
    return avgs

mavgs = moving_average(data_dept)
data_dept = data_dept.merge(mavgs,left_index=True,right_index = True)


# In[ ]:

#Dummyize Store 

for i in range(1,46):
    String = 'Store_'+ str(i)
    data_dept[String] = pd.Series(data_dept['Store'] == i, dtype=int)
data_dept.head()


# Dummyize Department 
for i in range(1,21):
    String = 'Dept'+ str(i)
    data_dept[String] = pd.Series(data_dept['Dept'] == i, dtype=int)
    
# Dummyize Season 
for i in range(1,5):
    String = 'Season' + str(i)
    data_dept[String] = pd.Series(data_dept['Season'] == i, dtype=int)
    
#Dummyize Year, Month of Sale
data_dept['Sales_Date'] =pd.to_datetime(data_dept.Sales_Date)
year_dummies = pd.get_dummies(data_dept.Sales_Date.dt.year,prefix='Year_')
month_dummies = pd.get_dummies(data_dept.Sales_Date.dt.month,prefix='Month_')
week_dummies = pd.get_dummies(data_dept.Sales_Date.dt.week,prefix='Week_')
data_dept = data_dept.merge(year_dummies,right_index=True,left_index=True)
data_dept = data_dept.merge(month_dummies,right_index=True,left_index=True)
data_dept = data_dept.merge(week_dummies,right_index=True,left_index=True)

#Dummyize Month-week of Sale:

# def week_of_month(dt):
#     """ Returns the week of the month for the specified date.
#     """
    
#     first_day = dt.replace(day=1)

#     dom = dt.day
#     adjusted_dom = dom + first_day.weekday()

#     return int(ceil(adjusted_dom/7.0))

# week_dummies = pd.get_dummies(data_dept.Sales_Date.apply(week_of_month),prefix='MonthWeek_')
# data_dept = data_dept.merge(week_dummies,right_index = True, left_index= True)


#Drop usless columns
dropCols = ['Season','Sales_Date','Store','Dept','Weekly_Sales']
data_dept= data_dept.drop(dropCols,1)

#Convert Target Variable to Float
data_dept['Target_Variable'] = data_dept['Target_Variable'].astype(float)


# In[ ]:

# Output the preprocessed data to a new file
data_dept.to_csv('processed_data_week.csv',index=False)

