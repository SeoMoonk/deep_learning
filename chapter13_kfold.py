#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tensorflow.keras import models, layers
from sklearn.model_selection import KFold


# In[2]:


df = pd.read_csv('data/sonar3.csv', header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# In[3]:


k = 5

kfold = KFold(n_splits=k, shuffle=True)

acc_score = []


# In[4]:


def model_fn():
    model = models.Sequential()
    model.add(layers.Dense(24, input_dim=60, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


# In[7]:


for train_index, test_index in kfold.split(X):
       X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
       y_train, y_test = y.iloc[train_index], y.iloc[test_index]
       
       model = model_fn()
       model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
       
       h = model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=0) # 학습
       # verbose의 default 값은 1인데, 0으로 설정하면 학습과정의 출력을 생략할 수 있다.
       
       accuracy = model.evaluate(X_test, y_test)[1] # 정확도 구하기
       acc_score.append(accuracy) # 정확도 리스트에 저장


# In[8]:


avg_acc_score = sum(acc_score) / k


# In[9]:


print('정확도 : ', acc_score)
print('정확도 평균 : ', avg_acc_score)


# In[ ]:




