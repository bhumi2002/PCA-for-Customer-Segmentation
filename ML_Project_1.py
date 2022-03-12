#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1.import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[54]:


#2.import the dataset
dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[55]:


y


# In[56]:


#3.split the dataset into test set and training set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.2,random_state=0)


# In[57]:


y_test


# In[58]:


#4.features scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[77]:


#5 applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


# In[60]:


#6 train logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)


# In[61]:


#7 confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)


# In[62]:


cm


# In[63]:


#8 accuracy of the model
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[73]:


#9 visualizing training set result
from matplotlib.colors import ListedColormap
x_set,y_set = x_train,y_train
x1,x2= np.meshgrid(np.arange(start = x_set[:,0].min() -1 ,
                          stop = x_set[:,0].max()+1, step = 0.01),
                np.arange(start = x_set[:,1].min()-1,
                          stop = x_set[:,1].max()+1, step = 0.01))


plt.contourf(x1 ,x2 ,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red' ,'green','blue')))

for i , j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set == j , 0] ,x_set[y_set == j ,1],
                c = ListedColormap(('red' ,'green','blue'))(i),label = j )
    
plt.title("Logistic Regression after PCA"),
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.show()


# In[65]:


x1


# In[75]:


from matplotlib.colors import ListedColormap
x_set,y_set = x_test,y_test
x1,x2= np.meshgrid(np.arange(start = x_set[:,0].min() -1 ,
                          stop = x_set[:,0].max()+1, step = 0.01),
                np.arange(start = x_set[:,1].min()-1,
                          stop = x_set[:,1].max()+1, step = 0.01))


plt.contourf(x1 ,x2 ,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red' ,'green','blue')))

for i , j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set == j , 0] ,x_set[y_set == j ,1],
                c = ListedColormap(('red' ,'green','blue'))(i),label = j )
    
plt.title("Logistic Regression after PCA[test Dataset]"),
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.show()


# In[ ]:




