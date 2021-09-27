#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn


# In[2]:


import pandas 
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


# In[3]:


url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names = names)


# In[4]:


print(dataset.shape)


# In[5]:


print(dataset.head(20))


# In[6]:


print(dataset.describe())


# In[7]:


print(dataset.groupby('class').size())


# In[8]:


dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()


# In[9]:


dataset.hist()
pyplot.show()


# In[10]:


scatter_matrix(dataset)
pyplot.show()


# In[11]:


array = dataset.values
x = array[:, 0:4]
y = array[:, 4]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state = 1)


# In[12]:


models = []
models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(("SVM", SVC(gamma='auto')))


# In[13]:


results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[14]:


pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


# In[16]:


model = SVC(gamma='auto')
model.fit(x_train, y_train)
predictions = model.predict(x_validation)


# In[20]:


print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))


# In[ ]:




