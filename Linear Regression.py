#!/usr/bin/env python
# coding: utf-8

# How does Linear Regression work, what are the steps do we follow, here is the code. 
# 
# Goal- Find a staright line that fits the data. 
# 
# 1. inorder to find the best staright line we need to test different straight lines 
# 2. but , we cannnot test all the possible straight lines as infinite number of straight lines are possible
# 3. so, we shrotlist some (1000) of the straights lines
# 4. And them among the shortlised lines, I will see which is the best one by looking at the "sum of squared residuals" values
# 5. The line having the lowest sum of squared rediduals will be the called the best staright line. 
# the equation of line y=mx+c
# here 
# y = Target
# m= slope
# x=feature
# c=y intercept
# 
# if we have 2 feature the line equation is 
# y=m1x1+m2x2+c and  it is 2 dimenstion. 
# 
# In regression model we see the performance of R Squared. R Squared range will be the -infinity to 1. as close to 1 will be the best performance
# 
# 

# In[4]:


#importing the liberies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df=pd.read_csv("C:/Users/USER/Desktop/Machine Learnng/Ml_bi_data/gapminder.csv")


# In[7]:


df


# In[8]:


df.isnull().sum()


# In[9]:


#here we dont have null values we can processed with the next step of extracting features and Targets


# In[15]:


target=df["fertility"]


# In[16]:


feature=df.drop("fertility",axis="columns")


# In[17]:


feature


# basically when we are buliding a model that model features should contain all are numeric in nature so here Region is not in numeric so we will convert that to numberic using using dummy encoding techqunie

# In[20]:


feature=pd.get_dummies(feature,columns=["Region"],drop_first=True)


# In[21]:


type(feature)


# In[22]:


feature.shape


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(feature,target,test_size=0.2,random_state=3)


# feature should be on same scale

# In[26]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[27]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[28]:


model.fit(x_train,y_train)


# In[29]:


model.score(x_test,y_test)


# Here 0.8 will be the r squared value. 

# In[ ]:




