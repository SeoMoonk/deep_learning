#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


# In[4]:


df = pd.read_csv('data/house_train.csv')

df.head()


# In[5]:


df.isnull().sum().sort_values(ascending=False).head(20)


# In[6]:


df = pd.get_dummies(df, dtype=int)

df.head()


# In[7]:


df = df.fillna(df.mean()) #결측치를 채워줌.


# In[8]:


df_corr = df.corr() # 데이터 사이의 상관관계를 저장
df_corr.head()


# In[9]:


df_corr_sort = df_corr.sort_values('SalePrice', ascending=False) 
# 집 값과 연관이 깊은것부터 정렬

df_corr_sort['SalePrice'].head(10)


# In[10]:


cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
       '1stFlrSF', 'FullBath', 'BsmtQual_Ex', 'TotRmsAbvGrd']

sns.pairplot(df[cols])
plt.show()


# In[11]:


cols_train = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 
              'TotalBsmtSF', '1stFlrSF']

X_train_pre = df[cols_train]
y = df['SalePrice'].values


# In[11]:


X_train, X_test, y_train, y_test = \
            train_test_split(X_train_pre, y, test_size=0.2)


# In[12]:


model = models.Sequential()
model.add(layers.Dense(10, input_dim = X_train.shape[1], activation='relu'))
model.add(layers.Dense(30, activation='relu'))
model.add(layers.Dense(40, activation='relu'))
model.add(layers.Dense(1)) #활성함수 사용 X
model.summary()

# 6 * 10 + 10 = 70
# 10 * 30 + 30 = 330
# 30 * 40 + 40 = 1240
# 40 * 1 + 1  = 41
# 1681


# In[13]:


#선형 회귀 이므로 손실함수는 평균 제곱 오차, metrix['accuracy']도 회귀에서는 빠짐.
model.compile(loss='mean_squared_error', optimizer='adam') 

early_stopping = EarlyStopping(monitor='val_loss', patience=20)

modelpath = "model/chapter14/Chapter14_house.hdf5"

checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
                             verbose=0, save_best_only=True)


# In[14]:


h = model.fit(X_train, y_train, validation_split=0.25, 
              epochs=2000, batch_size=32, callbacks=[early_stopping, checkpoint])

# 여기서의 손실은 가격 자체의 손실. (기존보다 크게 나온다.)


# In[16]:


real_price = []
pred_price = []
X_num = []

n_iter = 0
y_prediction = model.predict(X_test).flatten()

for i in range(50):
    real = y_test[i]
    prediction = y_prediction[i]
    print("실제 가격 : {:.2f}, 예상 가격 : {:.2f}".format(real, prediction))
    real_price.append(real)
    pred_price.append(prediction)
    n_iter += 1
    X_num.append(n_iter)


# In[17]:


plt.plot(X_num, pred_price, label='predicted price')
plt.plot(X_num, real_price, label='real price')
plt.legend()
plt.show()


# In[ ]:




