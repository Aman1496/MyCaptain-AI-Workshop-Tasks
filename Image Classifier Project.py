#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('mnist_test.csv')


# In[3]:


data.head()


# In[4]:


a = data.iloc[3,1:].values


# In[6]:


a = a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[7]:


df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state=4)


# In[9]:


y_train.head()


# In[11]:


rf = RandomForestClassifier(n_estimators=100)


# In[12]:


rf.fit(x_train, y_train)


# In[13]:


pred = rf.predict(x_test)


# In[14]:


pred


# In[16]:


s = y_test.values
count = 0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count+1


# In[17]:


count


# In[18]:


len(pred)


# In[19]:


1891/2000


# In[ ]:




