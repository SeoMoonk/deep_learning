import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/pima-indians-diabetes3.csv') #data-frame
df.head()
#df.tail() # default => 5

df.info()

print(type(df['diabetes'])) #특정 컬럼만 뽑아낼 수 있다.(시리즈로 출력 => [인덱스, 값] 쌍)

df['diabetes'].value_counts() #데이터가 어떻게 분포되어있는지 (0은 몇개고, 1은 몇갠지 ... 등 )

df.describe() #data-frame의 전체적인 통계 정보

df.corr() 
#각 항목별 상관관계를 대조? (ex => 나이가 많으면 당뇨일 확률, 보통 0.3이상이 연관있다고 봄. )

#df.corr colormap
colormap = plt.cm.gist_heat
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=colormap,
            linecolor='white', annot=True)
plt.show()

#plasma 항목에 대한 히스토그램
plt.hist(x=[df.plasma[df.diabetes==0], df.plasma[df.diabetes==1]], 
         bins=30, histtype='barstacked', label=['normal','diabetes'])
plt.legend()
plt.show()

from tensorflow.keras import models, layers

df.head()

X = df.iloc

X = df.iloc[:, :8]
y = df.iloc[:, 8]

print(type(X), type(y))

model = models.Sequential()
model.add(layers.Dense(12, input_dim=8, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
# (8 * 12) + 12
# (12 * 8) + 8
# (8 * 1) + 1
# 108 + 104 + 8 = 220

model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

h = model.fit(X, y, epochs=100, batch_size=5)

