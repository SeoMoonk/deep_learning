#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint

import os


# In[2]:


df = pd.read_csv('data/wine.csv', header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


# In[3]:


model = models.Sequential()

model.add(layers.Dense(30, input_dim=12, activation='relu'))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()


# In[4]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[5]:


MODEL_DIR = 'model/chapter14/'

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
    
modelpath = MODEL_DIR + '{epoch:02d}-{val_accuracy:.4f}.hdf5'

checkpoint = ModelCheckpoint(filepath=modelpath, verbose=1)


# In[6]:


h = model.fit(X_train, y_train, epochs=50, batch_size=500, validation_split=0.25,
             verbose=0, callbacks=[checkpoint])


# In[7]:


score = model.evaluate(X_test, y_test)
print('Test Accuracy : %.4f ' % (score[1]))


# In[8]:


del model

model = models.Sequential()

model.add(layers.Dense(30, input_dim=12, activation='relu'))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


h = model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25, verbose=0)


# In[12]:


hist_df = pd.DataFrame(h.history) # history 객체를 통해 학습 히스토리의 결과를 확인

hist_df.head()


# In[14]:


hist_df.info()


# In[19]:


import numpy as np

y_vloss = hist_df['val_loss']
y_loss = hist_df['loss']

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, 'o', c='red', markersize=2, label='Test_loss')
plt.plot(x_len, y_loss, 'o', c='blue', markersize=2, label='Train_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# In[ ]:




