# Databricks notebook source
import numpy as np
import pandas as pd
from datetime import time, tzinfo, timedelta
import datetime as datetime

from pyspark.sql import functions as f
from functools import partial
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline


### -----------------------------------------------------------------------------------------------------# 
### -----------------------------------------------------------------------------------------------------# 
#Goal is to get a dataframe that looks like this (will then be able to run it by product):
###Calc AUC for each period, period 

eff_date_period AUC  product(colname)
2022-07-01      0.5  sales 
2022-08-01      0.8  sales
2022-09-01      0.6  sales    

### -----------------------------------------------------------------------------------------------------# 
### -----------------------------------------------------------------------------------------------------# 




# COMMAND ----------

##EXAMPLE DF

data = {'month': ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'January', 'February', 'March'],
    'kpi': ['sales', 'sales quantity', 'sales', 'sales', 'sales', 'sales', 'sales', 'sales quantity', 'sales', 'sales', 'sales', 'sales'],
    'financial_year': [2022, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023],
    'prediction': [0.30, 0.20, 0.88, 0.10, 0.55, 0.33, 0.56, 0.99, 0.99, 0.75, 0.73, 0.61],
    'sub_renew': [0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0]
    }
df2 = pd.DataFrame(data)
df2['my'] = pd.to_datetime(df2['financial_year'].astype(str)  + df2['month'], format='%Y%B')
df2['my2'] = df2['my'].dt.strftime('%Y-%m-%d')
df2.head()

# COMMAND ----------

# COMMAND ----------


### Create a list of dates to loop over 


my = [*df2['my2'].drop_duplicates()]
my


### -----------------------------------------------------------------------------------------------------# 
### -----------------------------------------------------------------------------------------------------# 
# ATTEMPT 1: Add For Loop -Does not work 

def ck(product_):
# def ck(product_, dt_period_):
  temp=pd.DataFrame(columns=['eff_date_period','auc','product'])
  temp_lst = []
  
  for month_ in my:
      dftemp = df2.loc[(df2['kpi'] == product_)]

      X = dftemp['prediction'].to_numpy()
      y = dftemp['sub_renew'].to_numpy()

    #   print(X, '\n', y)
    #   print(type(X), '\n', type(y))

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)
      X_train = X_train.reshape(-1, 1)
      X_test = X_test.reshape(-1, 1)

      # Create the estimator - pipeline
      pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))

      # Create training test splits using all features
      pipeline.fit(X_train,y_train)
      probs = pipeline.predict_proba(X_test)
      fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1], pos_label=1)
      roc_auc = auc(fpr, tpr)


      #temp=pd.DataFrame(columns=['eff_date_period','auc','product'])
#       temp=temp.append({'eff_date_period': dt_period_,'auc':roc_auc,'product': product_},ignore_index=True)    
      temp=temp.append({'eff_date_period': month_,'auc':roc_auc,'product': product_},ignore_index=True)
      temp_lst = temp_lst.append(roc_auc)
      print(temp_lst)
      print(month_)
      
    #   return roc_auc
      return temp


# a_dte  = ck(product_ = "sales", dt_period_="2022-07-01")
x_dte  = ck(product_ = "sales")
# print(dte)
#   b = ck(product_ = "sales", dt_period_="2022-08-01")
#   c = ck(product_ = "sales", dt_period_="2022-09-01")

### -----------------------------------------------------------------------------------------------------# 
### -----------------------------------------------------------------------------------------------------# 



# ATTEMPT 2,This works, but still need to concatenate at the end. Don't want to have to do this, would prefer to use a loop
def ck(product_, dt_period_):
  dftemp = df2.loc[(df2['kpi'] == product_)]

  X = dftemp['prediction'].to_numpy()
  y = dftemp['sub_renew'].to_numpy()
  
#   print(X, '\n', y)
#   print(type(X), '\n', type(y))
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)
  X_train = X_train.reshape(-1, 1)
  X_test = X_test.reshape(-1, 1)

  # Create the estimator - pipeline
  pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))

  # Create training test splits using all features
  pipeline.fit(X_train,y_train)
  probs = pipeline.predict_proba(X_test)
  fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1], pos_label=1)
  roc_auc = auc(fpr, tpr)
  

  temp=pd.DataFrame(columns=['eff_date_period','auc','product'])
  temp=temp.append({'eff_date_period': dt_period_,'auc':roc_auc,'product': product_},ignore_index=True)                     
  
#   return roc_auc
  return temp #comb_df


a2 = ck(product_ = "sales", dt_period_="2022-07-01")
b = ck(product_ = "sales", dt_period_="2022-08-01")
c = ck(product_ = "sales", dt_period_="2022-09-01")
# a = ck(product_ = "sales", dt_period_="2022-08-01")

# COMMAND ----------


  






