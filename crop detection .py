#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("crop_recommendation.csv")

df.head()


# ## Preprocessing 

# In[2]:


X = df[df.columns[:-1]]
X


# In[3]:


y = df["label"]
y


# In[4]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


# In[5]:


Y = le.fit_transform(y)


# In[6]:


df["label"] = Y


# In[7]:


df


# ## Analysing the data

# In[9]:


profile= ProfileReport(df)


# In[10]:


profile.to_file("reports.html")


# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=145)


# ## Training the model with multiple model and finding the accuracy

# In[12]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

# create instances of all models
models = {
    'Logistic Regression': LogisticRegression(solver="liblinear"),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
}

from sklearn.metrics import accuracy_score
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'{name}:\nAccuracy: {acc:.4f}')
    
    


#  On training the different models we can clearly see the random forest is the best model for this problem statement 
# 

# ## Training the model using Random forest classifier

# In[13]:


from sklearn.ensemble import RandomForestClassifier


# In[14]:


rdf = RandomForestClassifier()
rdf.fit(X_train.to_numpy(),y_train)


# In[15]:


from sklearn.metrics import accuracy_score

y_pred = rdf.predict(X_test.to_numpy())
print(accuracy_score(y_test,y_pred))


# ## Saving the model to file for later use

# In[16]:


import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(rdf, f)


# ## Loading the model

# In[17]:


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

model


# In[20]:


y_pred = model.predict(X_test.to_numpy())
print(accuracy_score(y_test,y_pred))


# In[21]:


y_pred = model.predict([[90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536]], )
y_pred

pred_label = le.inverse_transform(y_pred)[0]
pred_label


# In[22]:


np.save('classes.npy', le.classes_)


# In[23]:


encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy', allow_pickle=True)


# In[24]:


pred_label = encoder.inverse_transform(y_pred)[0]
pred_label


# In[ ]:




