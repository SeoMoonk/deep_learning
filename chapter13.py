#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tensorflow.keras import models, layers

# pandas의 read_csv 함수는 무조건 첫 열을 컬럼명으로 인식한다. 
# 이를 방지하기 위해서 첫 열이 컬럼명이 아닐 경우, header = None 옵션을 사용해야 한다.
df = pd.read_csv('data/sonar3.csv', header=None)
df.head()


# In[2]:


# 보통 자료의 마지막 컬럼에는 최종 결과(?) 가 들어있는데, value_counts를 통해
# 결과가 어떻게 분류되어 있는지 확인할 수 있다. 여기서는 ( 광물 / 광물X )가 나뉘어진 상태.
df[60].value_counts()


# In[3]:


X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# In[4]:


model = models.Sequential()
model.add(layers.Dense(24, input_dim=60, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# 60 * 24 + 24 = 1464
# 24 * 10 + 10 = 250
# 10 * 1 + 1 = 11


# In[10]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[11]:


X.shape


# In[12]:


y.shape


# In[13]:


h = model.fit(X, y, epochs=200, batch_size=5)

# 정말 확률이 100%? => 과적합.
# 훈련용 데이터와 테스트용 데이터를 나눠야 한다.


# In[ ]:




