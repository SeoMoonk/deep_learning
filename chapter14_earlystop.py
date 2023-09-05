#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

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

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[4]:


early_stopping = EarlyStopping(monitor='val_loss', patience=20)

modelpath = 'model/chapter14/wine_model.hdf5' #이전보다 결과가 좋아질 때만 저장

checkpoint = ModelCheckpoint(filepath=modelpath, monitor='var_loss',
                            verbose = 0, save_best_only=True)


# In[5]:


h = model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25,
             verbose=1, callbacks=[early_stopping, checkpoint])


# In[6]:


score = model.evaluate(X_test, y_test)
print('Test Accuracy : ', score[1])


# In[ ]:




