#!/usr/bin/env python
# coding: utf-8

#    ## Iris Veri Kümesinde Naive Bayes Sınıflandırma Modeli Çalıştırılması

# ### About Iris Dataset
# The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository.
# 
# It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.
# 
# The columns in this dataset are:
# 
# Id,
# SepalLengthCm,
# SepalWidthCm,
# PetalLengthCm,
# PetalWidthCm,
# Species

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv("iris.csv")
df.head()


# In[2]:


#Id özniteliğini kaldıralım
df = df.drop('Id', axis = 1)


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


#Eksik veri var mı?
df.isnull().sum()


# In[7]:


#Sınıf dağılımları nasıl?
df.groupby("Species").size()


# In[9]:


#Histogram grafiği incelemesi
df.hist()


# In[10]:


#Kutu çizim grafiği incelemesi
#df.plot(kind='box', sharex=False, sharey=False)
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)


# In[11]:


df.corr()


# In[13]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

#Eğitim için ilgili öznitlelik değerlerini seç
X = df.iloc[:, :-1].values

#Sınıflandırma öznitelik değerlerini seç
Y = df.iloc[:, -1].values


# In[14]:


#Eğitim ve doğrulama veri kümelerinin ayrıştırılması
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[15]:


#Naive Bayes modelinin oluşturulması
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[16]:


#NB modelinin K-katlamalı çapraz doğrulama ile ACC değerinin hesaplanması
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)


# In[17]:


cv_results


# In[18]:


msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
msg


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))


# ## Iris Veri Kümesinde K-Means Çalıştırılması

# In[20]:


#Öncelikle kategorik sınıf değerini sayısal forma çevirelim
df["Species"] = df["Species"].astype('category').cat.codes
df.head(10)


# In[21]:


#Y değerini sayısal haliyle yeniden ayarlayalım
Y = df.iloc[:, -1].values


# In[22]:


#Sınıf dağılımlarını tekrar kontrol edelim
df.groupby("Species").size()


# ### K-Means çalıştırma adımı

# In[23]:


from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, n_jobs = 4, random_state=21)
km.fit(X)


# In[24]:


centers = km.cluster_centers_
centers


# In[25]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], X[:, 3], c=Y)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], centers[:, 3], marker='*', c='#050505', s=1000)


# In[ ]:




