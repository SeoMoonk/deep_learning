from tensorflow.keras import models, layers
import pandas as pd
import tensorflow as tf

df = pd.read_csv('data/pima-indians-diabetes3.csv') #data-frame

X = df.iloc[:, :8]
y = df.iloc[:, 8]

model_name = 'pima_indian_diabetes'
batch_size = 5

params = {
    'model_name': model_name,
    'input_dim': 8,
    'hidden_dim_01': 12,
    'hidden_dim_02': 8,
    'output_dim': 1
}

class PimaIndian(tf.keras.Model):
    def __init__(self, **kargs):
        super(PimaIndian, self).__init__(name=kargs['model_name'])
        self.fc1 = layers.Dense(kargs['hidden_dim_01'],
                               input_dim = kargs['input_dim'], activation = 'relu')
        self.fc2 = layers.Dense(kargs['hidden_dim_02'], activation = 'relu')
        self.fc3 = layers.Dense(kargs['output_dim'], activation = 'sigmoid')
        
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = PimaIndian(**params)

model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

h = model.fit(X, y, epochs=100, batch_size=5)



