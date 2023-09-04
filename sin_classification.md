```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
X = np.arange( -1.0, 1.1, 0.1)
Y = np.arange( -1.0, 1.1, 0.1)
```


```python
input_data = []
correct_data = []

for x in X:
    for y in Y:
        input_data.append([x, y])
        if y < np.sin(np.pi * x):
            correct_data.append([0, 1])
        else:
            correct_data.append([1, 0])
```


```python
n_data = len(correct_data)
input_data = np.array(input_data)
correct_data = np.array(correct_data)
```


```python
n_in = 2
n_mid = 6
n_out = 2
```


```python
wb_width = 0.01
eta = 0.05
epoch = 101
interval = 20
```


```python
class MiddleLayer:
    def __init__(self, n_upper, n): #n_upper : 입력의 수, n : 출력의 수
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.randn(n)
    
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = 1 / ( 1 + np.exp(-u))
        
    def backward(self, grad_y):
        delta = grad_y * (1 - self.y) * self.y
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta)
        self.grad_x = np.dot(delta, self.w.T)
        
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
```


```python
class OutputLayer:
    def __init__(self, n_upper, n): #n_upper : 입력의 수, n : 출력의 수
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.randn(n)
        
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u) / np.sum(np.exp(u), axis=1, keepdims=True)
        
    def backward(self, grad_t):
        delta = self.y - t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta)
        self.grad_x = np.dot(delta, self.w.T)
        
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
```


```python
middle_layer = MiddleLayer(n_in, n_mid)
output_layer = OutputLayer(n_mid, n_out)
```


```python
sin_data = np.sin(np.pi * X)
```


```python
for i in range(epoch):
    index_random = np.arange(n_data)
    np.random.shuffle(index_random)
    
    total_error = 0
    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []
    
    for idx in index_random:
        x = input_data[idx]
        t = correct_data[idx]
        
        middle_layer.forward(x.reshape(1, 2))
        output_layer.forward(middle_layer.y)
        
        output_layer.backward(t.reshape(1, 2))
        middle_layer.backward(output_layer.grad_x)
        
        middle_layer.update(eta)
        output_layer.update(eta)
        
        if i % interval == 0:
            y = output_layer.y.reshape(-1)
            total_error += -np.sum(t * np.log(y + 1e-7))
            
            if y[0] > y[1]:
                x_1.append(x[0])
                y_1.append(x[1])
            else:
                x_2.append(x[0])
                y_2.append(x[1])
                
    if i % interval == 0:
        
        # 출력 그래프 표시
        plt.plot(X, sin_data, linestyle='dashed')
        plt.scatter(x_1, y_1, marker='+')
        plt.scatter(x_2, y_2, marker='x')
        plt.show()
        
        # 에포크 수와 오차 표시
        print("Epoch:" + str(i) + "/" + str(epoch), "Error:" + str(total_error/n_data))
        
        
```

![output_10_0](https://github.com/SeoMoonk/deep_learning/assets/39723465/fcde4b0c-98ce-4246-8e82-9a73cb6ffe01)


![output_10_2](https://github.com/SeoMoonk/deep_learning/assets/39723465/c117c5ed-a955-4d32-a21f-8915e726b688)


![output_10_4](https://github.com/SeoMoonk/deep_learning/assets/39723465/fd15569c-ce07-474d-a768-363d219973eb)


![output_10_6](https://github.com/SeoMoonk/deep_learning/assets/39723465/4b760409-0e3b-4319-aa3f-298700385e84)


![output_10_8](https://github.com/SeoMoonk/deep_learning/assets/39723465/348fb205-33ce-42e0-ae78-52ea4b20b77d)


![output_10_10](https://github.com/SeoMoonk/deep_learning/assets/39723465/68a77b8a-4542-46d8-83aa-ed4aaee2f850)


```python

```
