#!/usr/bin/env python
# coding: utf-8

# In[ ]: Importing Necessary Modules
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle


# In[ ]: Loading and viewing the data
ghana_data = pd.read_csv('ghana_census_data.csv')
ghana_data.sample(5)


# In[ ]: Remove data fields that are not numbers
ghana_data = ghana_data.dropna()
ghana_data.shape


# In[ ]: Get an overview of the statistic on the data
ghana_data.describe()


# In[ ]: Visualizing the Data
fig, ax = plt.subplots(figsize=(12, 8))

plt.scatter(ghana_data['year'], ghana_data['population'])
plt.xlabel("Year")
plt.ylabel("Population")


# In[198]:
ghana_data_corr = ghana_data.corr()
ghana_data_corr


# In[199]:
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(ghana_data_corr, annot=True)


# In[200]:
X = ghana_data.drop('population', axis=1)
Y = ghana_data['population']


# In[201]:
X.columns


# In[202]:
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[203]:
x_train.shape, x_test.shape


# In[204]:
y_train.shape, y_test.shape


# In[205]:
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)


# In[206]:
print("Training_score : ", linear_model.score(x_train, y_train))


# In[207]:
predictors = x_train.columns
predictors


# In[208]:
coef = pd.Series(linear_model.coef_, predictors).sort_values()
print(coef)


# In[233]:
y_pred = linear_model.predict(x_test)


# In[217]:
df_pred_actual = pd.DataFrame({'predicted': y_pred, 'actual': y_test})

df_pred_actual


# In[211]:
from sklearn.metrics import r2_score
print('Testing_score : ', r2_score(y_test, y_pred))


# In[213]:
fig, ax = plt.subplots(figsize=(12, 8))

plt.scatter(y_pred, y_test)
plt.show()


# In[218]:
df_pred_actual_sample = df_pred_actual.sample(12)
df_pred_actual_sample = df_pred_actual_sample.reset_index()


# In[220]:
df_pred_actual_sample.head()


# In[222]:
plt.figure(figsize=(20, 10))

plt.plot(df_pred_actual_sample['predicted'], label='Predicted')
plt.plot(df_pred_actual_sample['actual'], label='Actual')

plt.ylabel('Population Census')

plt.legend()
plt.show()


# In[ ]:
pickle.dump(linear_model, open('model.pkl','wb'))


# In[ ]:
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2002]]))

# %%
