# Databricks notebook source
import numpy as np
import pandas as pd
from datetime import time, tzinfo, timedelta
import datetime as datetime

from pyspark.sql import functions as f
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

# COMMAND ----------

data = {'month': ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'January', 'February', 'March'],
    'kpi': ['sales', 'sales quantity', 'sales', 'sales', 'sales', 'sales', 'sales', 'sales quantity', 'sales', 'sales', 'sales', 'sales'],
    'financial_year': [2022, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023],
    'prediction': [0.30, 0.20, 0.88, 0.10, 0.55, 0.33, 0.56, 0.99, 0.99, 0.75, 0.73, 0.61],
    'sub_renew': [0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0]
    }
df2 = pd.DataFrame(data)
df2['my'] = pd.to_datetime(df2['financial_year'].astype(str)  + df2['month'], format='%Y%B')
df2.head()

# COMMAND ----------

def ck(product_, dt_period_):
  dftemp = df2.loc[(df2['kpi'] == product_)]
  dftemp.head()
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
  
  comb_df = pd.DataFrame({'Eff_date_period' : '1900-01-01','AUC': 99, 'colname' : 'a'}, index=[0])
  #comb_df = pd.concat([comb_df,pd.DataFrame({'Eff_date_period' : [dt_period_],'AUC': [roc_auc], 'colname' : [product_]})], ignore_index=True)
  
  ##another attempt
  comb_df.loc[len(comb_df)] = [dt_period_, roc_auc, product_]
  comb_df.head()
  
  
  return comb_df.head()


ck(product_ = "sales", dt_period_="2022-07-01")
ck(product_ = "sales", dt_period_="2022-08-01")

# COMMAND ----------


###Goal is to return results in a dataframe that looks like:

product dt_period roc_auc
sales   2022-07-01 0.7
sales   2022-08-01 0.8
sales   2022-09-01 0.9 




