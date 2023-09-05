#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('data/sonar3.csv', header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)


# In[4]:


model = models.Sequential()
model.add(layers.Dense(24, input_dim=60, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()


# In[5]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[6]:


h = model.fit(X_train, y_train, epochs=200, batch_size=5)


# In[7]:


score = model.evaluate(X_test, y_test) #evaluate -> 정확도 측정


# In[8]:


score


# In[11]:


print('Test Accuracy:', score[1])


# In[12]:


model.save('data/sonar.hdf5')  #테스트셋 저장


# In[13]:


del model


# In[15]:


model = models.load_model('data/sonar.hdf5') #저장해둔 모델을 삭제하고, 다시 불러오기


# In[20]:


#evaluate -> 정확도 측정 (파라미터 학습 x, 단순히 정확도 계산)
score = model.evaluate(X_test, y_test) 
print('Test Accuracy:', score[1])

# 객관적으로 모델의 성능을 측정하려면, 학습 데이터를 테스트 데이터로 사용해서는 안된다.


# In[ ]:


# 학습을 위해서는 데이터가 많아야 하는데, 항상 데이터가 많을 순 없을 것이다.
# 이 떄, 사용할 수 있는 방법 중 하나가 K겹 교차 검증이다.
# 데이터 셋을 다섯 개로 나눈 후, 4개는 학습, 1개는 테스트 셋으로 만들어 5번의 학습을 실시

