#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('data/iris3.csv')
df.head()


# In[3]:


df.info()


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


sns.pairplot(df, hue='species')
plt.show()


# In[6]:


X = df.iloc[:, :-1] # :-1은 마지막까지(length), -1은 마지막 하나
y = df.iloc[:, -1]


# In[7]:


X[:5]


# In[8]:


y[:5]


# In[9]:


y.shape


# In[10]:


# y = pd.get_dummies(y) 
# #기존데이터에서는 꽃의 종류에 따라 문자열(object)로 되있었는데,
# #후처리 함으로써 종류에 따라 True, False로 구분되도록 되었음. 
# #(근데, 필요한건 사실 0 or 1 (boolean) 값임) => 버전업 되었음
# # ===> dtype에 int를 주면 해결 가능

y = pd.get_dummies(y, dtype=int)


# In[11]:


y[:5]


# In[10]:


# from sklearn.preprocessing import LabelEncoder
# #pandas의 get_dummies 대신 sckit-Learning의 LabelEncoder 사용

# e = LabelEncoder()
# e.fit(y)
# y = e.transform(y)
# y.shape


# In[11]:


# y[:5] #종에 따라 (0, 1, 2)로 분류되었음.


# In[12]:


# y[-5:]


# In[13]:


# from tensorflow.keras.utils import to_categorical

# y = to_categorical(y)
# y.shape


# In[14]:


# y[:5]


# In[12]:


from tensorflow.keras import models, layers


# In[13]:


model = models.Sequential()
model.add(layers.Dense(12, input_dim=4, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.summary()

# 4 * 12 + 12 = 60
# 12 * 8 + 8 = 104
# 8 * 3 + 3 = 27


# In[16]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[17]:


h = model.fit(X, y, epochs=30, batch_size=5)


# In[ ]:




