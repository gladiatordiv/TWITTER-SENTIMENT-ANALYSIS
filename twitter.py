#!/usr/bin/env python
# coding: utf-8

# In[6]:





# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('train.csv')


# In[4]:


df


# In[5]:


df['label'].value_counts()


# In[6]:


df.info()


# In[7]:


ps=PorterStemmer()
corpus=[]
for i in range(len(df)):
    print(i)
    rev=re.sub("[^a-zA-Z]",' ',df['tweet'][i])
    rev=rev.lower()
    rev=rev.split()
    rev=[ps.stem(word) for word in rev if word not in set(stopwords.words("english"))]
    rev=' '.join(rev)
    corpus.append(rev)


# In[8]:


corpus


# In[9]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
x=cv.fit_transform(corpus).toarray()


# In[10]:


x


# In[11]:


x.shape


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,df['label'],test_size=0.20,random_state=0)


# In[13]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[14]:


from sklearn.naive_bayes import MultinomialNB


# In[15]:


mnb=MultinomialNB().fit(x_train,y_train)


# In[16]:


pred=mnb.predict(x_test)


# In[17]:


pred


# In[18]:


y_test


# In[19]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[20]:


print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))


# In[21]:


fnl=pd.DataFrame(np.c_[y_test,pred],columns=['Actual','Predicted'])
fnl


# In[22]:


import joblib


# In[23]:


joblib.dump(mnb,'Twitter_Sentiment_Analysis.pkl')


# In[24]:


model=joblib.load('Twitter_Sentiment_Analysis.pkl')


# In[25]:


def test_model(test_sentence):
    rev=re.sub("[^a-zA-Z]",' ',test_sentence)
    rev=rev.lower()
    rev=rev.split()
    rev=[ps.stem(word) for word in rev if word not in set(stopwords.words("english"))]
    rev=' '.join(rev)
    rev=cv.transform([rev]).toarray()
    output=model.predict(rev)[0]
    if output==0:
        print("Positive Analysis")
    else:
        print("Negative Analysis")


# In[26]:


test_model("this is very interesting post")


# In[ ]:





# In[ ]:




