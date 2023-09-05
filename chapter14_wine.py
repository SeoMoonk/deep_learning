#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('data/wine.csv', header=None)


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


# In[7]:


model = models.Sequential()

model.add(layers.Dense(30, input_dim=12, activation='relu'))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# 12 * 30 + 30 = 390
# 30 * 12 + 12 = 372
# 12 * 8 + 8 = 104
# 8 * 1 + 1 = 9
# => 875


# In[15]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[16]:


h = model.fit(X_train, y_train, epochs=50, batch_size=500, validation_split=0.25)
# 학습 셋의 25%를 검증 셋으로 사용하여 학습


# In[17]:


score = model.evaluate(X_test, y_test)
print('Test accuracy : ', score[1])


# In[ ]:




