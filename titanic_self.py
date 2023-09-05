#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


# In[3]:


df = pd.read_csv('data/titanic.csv')


# In[5]:


df.head()


# In[6]:


df.isnull().sum().sort_values(ascending=False).head(20)


# In[7]:


df = pd.get_dummies(df, dtype=int)

df.head()


# In[8]:


df = df.fillna(df.mean())


# In[9]:


df_corr = df.corr()
df_corr.head()


# In[10]:


df_corr_sort = df_corr.sort_values('Survived', ascending=False)

df_corr_sort['Survived'].head(10)


# In[13]:


cols = ['Survived', 'Sex_female', 'Fare', 'Embarked_C', 'Ticket_113760', 'Cabin_B96 B98',
       'Ticket_2666', 'Parch', 'Ticket_347742', 'Ticket_29106']

sns.pairplot(df[cols])
plt.show()


# In[14]:


cols_train = [ 'Sex_female', 'Fare', 'Embarked_C', 'Cabin_B96 B98', 'Parch']

X_train_pre = df[cols_train]
y = df['Survived'].values


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.2)


# In[16]:


model = models.Sequential()
model.add(layers.Dense(10, input_dim = X_train.shape[1], activation='relu'))
model.add(layers.Dense(30, activation='relu'))
model.add(layers.Dense(40, activation='relu'))
model.add(layers.Dense(1)) #활성함수 사용 X
model.summary()


# In[18]:


model.compile(loss='mean_squared_error', optimizer='adam') 

early_stopping = EarlyStopping(monitor='val_loss', patience=20)

modelpath = "model/chapter14/titanic.hdf5"

checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
                             verbose=0, save_best_only=True)


# In[19]:


h = model.fit(X_train, y_train, validation_split=0.25, 
              epochs=2000, batch_size=32, callbacks=[early_stopping, checkpoint])

# 여기서의 손실은 가격 자체의 손실. (기존보다 크게 나온다.)


# In[21]:


real_survive = []
pred_survive = []
X_num = []

n_iter = 0
y_prediction = model.predict(X_test).flatten()

for i in range(50):
    real = y_test[i]
    prediction = y_prediction[i]
    print("실제 생존 확률 : {:.2f}, 예상 생존 확률 : {:.2f}".format(real, prediction))
    real_survive.append(real)
    pred_survive.append(prediction)
    n_iter += 1
    X_num.append(n_iter)


# In[22]:


plt.plot(X_num, pred_survive, label='predicted survive')
plt.plot(X_num, real_survive, label='real survive')
plt.legend()
plt.show()


# In[ ]:




