from tensorflow.keras import models, layers
import numpy as np

data_set = np.loadtxt("./data/ThoraricSurgery3.csv", delimiter=",")
data_set

X = data_set[:, :16]
y = data_set[:, 16]

X[:3]

y[:3]

model = models.Sequential()
model.add(layers.Dense(30, input_dim=16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid' ))

model.summary() #510 = (16 * 30) + 30 => (16 * input)  + (bias)

model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

h = model.fit(X, y, epochs=5, batch_size=16)
