#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Title of Project
# Customer Churn Prediction


# In[3]:


# Objective
# The objective of this project is to predict customer churn in a bank using machine learning techniques. Churn prediction helps in identifying customers who are likely to leave the bank, allowing proactive retention strategies to be implemented.


# In[4]:


# Data Source
# The dataset used for this project is sourced from a CSV file named 'Churn_Modelling.csv', which contains information about bank customers.


# In[5]:


# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


# In[9]:


# Import Data
dataset = pd.read_csv('Churn_Modelling.csv')
dataset


# In[10]:


# Describe Data
dataset.info()
dataset.describe()


# In[14]:


# Data Visualization
dataset['Exited'].plot.hist()
dataset.drop(columns='Exited').corrwith(dataset['Exited'], numeric_only=True).plot.bar(figsize=(16, 9), title='Correlated with Exited Column', rot=45, grid=True)
corr = dataset.corr(numeric_only=True)
plt.figure(figsize=(16, 9))
sns.heatmap(corr, annot=True)


# In[ ]:


# Data Preprocessing
dataset = dataset.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
dataset = pd.get_dummies(data=dataset, drop_first=True)
dataset


# In[17]:


# Define Target Variable (y) and Feature Variables (X)
X = dataset.drop(columns='Exited')
y = dataset['Exited']


# In[18]:


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[27]:


# Modeling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train

# Logistic Regression Model
clf_lr = LogisticRegression(random_state=0).fit(X_train, y_train)

# Random Forest Classifier Model
clf_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train)


# In[21]:


# Model Evaluation
# Logistic Regression Results
y_pred_lr = clf_lr.predict(X_test)
results_lr = pd.DataFrame([['Logistic regression', accuracy_score(y_test, y_pred_lr), f1_score(y_test, y_pred_lr),
                            precision_score(y_test, y_pred_lr), recall_score(y_test, y_pred_lr)]],
                          columns=['Model', 'Accuracy', 'F1', 'Precision', 'Recall'])

# Random Forest Results
y_pred_rf = clf_rf.predict(X_test)
results_rf = pd.DataFrame([['Random Forest Classifier', accuracy_score(y_test, y_pred_rf), f1_score(y_test, y_pred_rf),
                            precision_score(y_test, y_pred_rf), recall_score(y_test, y_pred_rf)]],
                          columns=['Model', 'Accuracy', 'F1', 'Precision', 'Recall'])

# Display Results
results_concat = pd.concat([results_lr, results_rf], ignore_index=True)
results_concat


# In[23]:


# Prediction
single_obs = [[647, 40, 3, 85000.45, 2, 0, 0, 92012.45, 0, 1, 1]]
prediction = clf_rf.predict(scaler.fit_transform(single_obs))
print(prediction)


# In[24]:


# Explanation
"""The project involves predicting customer churn in a bank using two models: Logistic Regression and Random Forest Classifier. The dataset is preprocessed, features are scaled, and models are trained. Evaluation metrics such as accuracy, F1 score, precision, and recall are calculated for both models. Finally, the trained Random Forest model is used to predict churn for a new observation."""


# In[ ]:




