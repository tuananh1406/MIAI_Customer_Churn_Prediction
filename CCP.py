#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# EXP 01: Ko Upsampling, Ko Scale. Model: XGBOOST, RFC, GRADIENTBOOSTING
# EXP 02: Ko Upsampling, Scale. Model : Log, Xgboost, Rfc, Grad
# Exp 03: Up Samplong bằng SMOTE, ko scale: XGBOOST, RFC, GRADIENTBOOSTING
# Exp 04: Up Sampling, Scale: Log, XGB, RFC, Grad

# Exp 01:  Upsampling, Scale: Thử với log
# Exp 2: Không Upsampling, Ko scale: thử với XgBoost


# # Setup

# In[1]:


# get_ipython().system('pip install xgboost')
# get_ipython().system('pip install imbalanced-learn')
# get_ipython().system('pip install category_encoders')


# In[2]:


# Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
import category_encoders as ce
from imblearn.over_sampling import SMOTE, SVMSMOTE


# # Đọc dữ liệu

# In[4]:


customer_data = pd.read_csv("train.csv")


# In[5]:


customer_data


# # Xem qua dữ liệu -EDA

# In[ ]:


customer_data.info()


# In[ ]:


customer_data.isnull().sum()


# In[8]:


object_cols = [f for f in customer_data.columns if customer_data[f].dtype =="O"]
print(object_cols)


# In[ ]:


# for col in object_cols:
#   customer_data[col].value_counts().plot(kind='bar', figsize=(15,5))
#   plt.title(col)
#   plt.show()


# In[11]:


numeric_cols = [f for f in customer_data.columns if customer_data[f].dtype !="O"]
print(numeric_cols)


# In[ ]:


# for col in numeric_cols:
#   customer_data[col].hist()
#   plt.title(col)
#   plt.show()


# In[ ]:


# for col in numeric_cols:
#   customer_data.boxplot(column=[col])
#   plt.title(col)
#   plt.show()


# # Tiên xử lý dữ liệu để đưa vào model

# In[15]:


dataset = customer_data.copy()


# In[ ]:


he = ce.HashingEncoder(cols='state')
dataset_hash = he.fit_transform(dataset)
dataset_hash


# In[ ]:


dataset_hash_dummy = pd.get_dummies(dataset_hash, drop_first=True)
dataset_hash_dummy


# In[ ]:


# View correlation
corr = dataset_hash_dummy.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[22]:


# Remove correlation columns
dataset_hash_dummy_drop_corr = dataset_hash_dummy.drop(columns=["voice_mail_plan_yes","total_day_charge","total_eve_charge","total_night_charge","total_intl_charge"])


# # Exp 01: Upsampling = SMOTE, Scale = MINMAX và thử với Logistic

# In[26]:


# SMOTE & Scale
X = dataset_hash_dummy_drop_corr.drop(["churn_yes"],axis=1)
y = dataset_hash_dummy_drop_corr['churn_yes']

# Chia train ,test
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state=42)

# Upsampling = SMOTE
sm = SMOTE(k_neighbors=5)
X_train_resample, y_train_resample = sm.fit_resample(X_train,y_train)

#Scale

scale_columns = ['account_length', 'number_vmail_messages', 'total_day_minutes',
       'total_day_calls', 'total_eve_minutes', 'total_eve_calls',
       'total_night_minutes', 'total_night_calls', 'total_intl_minutes',
       'total_intl_calls', 'number_customer_service_calls']



scaler = MinMaxScaler()
scaler.fit(X_train_resample[scale_columns])
X_train_resample[scale_columns] = scaler.transform(X_train_resample[scale_columns])
X_test[scale_columns] = scaler.transform(X_test[scale_columns])


# In[28]:


# Logistic Regression
model_log = LogisticRegression() 
model_log.fit(X_train_resample, y_train_resample)
y_pred = model_log.predict(X_test)

# In ra du lieu
print(classification_report( y_test, y_pred))

plot_confusion_matrix(model_log, X_test, y_test)


# # Exp 2: XGBOOST, Ko Upsampling, Ko Scale

# In[29]:


# SMOTE & Scale
X = dataset_hash_dummy_drop_corr.drop(["churn_yes"],axis=1)
y = dataset_hash_dummy_drop_corr['churn_yes']

# Chia train ,test
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state=42)


# In[30]:


# XGBoost

import xgboost as xgb

model_xgb = xgb.XGBClassifier(random_state=42, n_estimators = 200)
model_xgb.fit(X_train, y_train)


y_pred = model_xgb.predict(X_test)
# In bao cao ket qua
print(classification_report( y_test, y_pred))
plot_confusion_matrix(model_xgb, X_test, y_test) 



# # Submit ket qua len Kaggle

# In[41]:


test = pd.read_csv("test.csv")
id_submit = test['id']


# In[42]:


test.drop(columns=['id'], inplace=True)


# In[43]:


test_hash_state = he.fit_transform(test)
test_hash_state.head()


# In[44]:


test_dummy =  pd.get_dummies(test_hash_state,drop_first=True)
test_dummy_drop_corr = test_dummy.drop(columns=["voice_mail_plan_yes","total_day_charge","total_eve_charge","total_night_charge","total_intl_charge"])


# In[45]:


test_dummy_drop_corr.columns


# In[46]:


y_pred_submit = model_xgb.predict(test_dummy_drop_corr)


# In[47]:


y_pred_submit


# In[48]:


submit_result = pd.DataFrame({'id': id_submit,'churn': y_pred_submit})
submit_result


# In[49]:


submit_result.churn.replace([0,1],['no','yes'],inplace=True)
submit_result


# In[51]:


submit_result.to_csv("miai_submit.csv", index=False)

