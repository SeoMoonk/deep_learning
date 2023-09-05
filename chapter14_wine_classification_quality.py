#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler


# In[3]:


df = pd.read_csv('data/wine.csv', header=None)


# In[4]:


df['quality'] = df[11] # quality 속성을 11을 복사해서 만듬.


# In[5]:


del df[11] # 11은 삭제


# In[6]:


df['quality'].value_counts()


# In[7]:


X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# In[8]:


y = y.values


# In[9]:


y


# In[10]:


print(type(y))


# In[11]:


#tensorflow.keras.utils.to_categorical -> numpy 에서 one hot encoding 으로 quality를 변환.
y = to_categorical(y, num_classes=11, dtype=int)


# In[12]:


y.shape


# In[13]:


y[:3]


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


# In[17]:


min_max = MinMaxScaler()
X_scaled_train = min_max.fit_transform(X_train)


# In[18]:


X_scaled_train[:3]


# In[19]:


X_scaled_test = min_max.transform(X_test)


# In[21]:


X_scaled_test[:3]


# In[22]:


model = models.Sequential()
model.add(layers.Dense(30, input_dim=12, activation='relu')) # 12 * 30 + 30 = 390
model.add(layers.Dense(24, activation='relu')) # 30 * 24 + 24 = 744
model.add(layers.Dense(11, activation='softmax')) # 24 * 11 + 11 = 275
model.summary()


# In[23]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[24]:


h = model.fit(X_scaled_train, y_train, epochs=30, batch_size=20)


# In[25]:


score = model.evaluate(X_scaled_test, y_test)
print('Test ccuracy : %.4f ' % (score[1]))


# In[ ]:


# ( x_uk = x_k - x_min / x_max - x_min ) 
# 모든 데이터의 값을 0 ~ 1 사이의 값으로 정규화 => "MinMaxScaler"

