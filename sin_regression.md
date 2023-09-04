```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
input_data = np.arange(0, np.pi * 2, 0.1)
correct_data = np.sin(input_data)
input_data = (input_data - np.pi) / np.pi # -1.0 ~ 1.0 으로 input_data가 바뀜.
n_data = len(correct_data)
```


```python
n_in = 1
n_mid = 3
n_out = 1
```


```python
wb_width = 0.01
eta = 0.1
epoch = 2001
interval = 200
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
        self.y = u
        
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
for i in range(epoch):
    index_random = np.arange(n_data)
    np.random.shuffle(index_random)
    
    total_error = 0
    plot_x = []
    plot_y = []
    
    for idx in index_random:
        x = input_data[idx]
        t = correct_data[idx]
        
        middle_layer.forward(x.reshape(1, 1))
        output_layer.forward(middle_layer.y)
        
        output_layer.backward(t.reshape(1, 1))
        middle_layer.backward(output_layer.grad_x)
        
        middle_layer.update(eta)
        output_layer.update(eta)
        
        if i % interval == 0:
            y = output_layer.y.reshape(-1)
            total_error += 1.0 / 2.0 * np.sum(np.square(y - t))
            
            plot_x.append(x)
            plot_y.append(y)
            
    if i%interval == 0:
        
        # 출력 그래프 표시
        plt.plot(input_data, correct_data, linestyle="dashed")
        plt.scatter(plot_x, plot_y, marker="+")
        plt.show()
        
        # 에포크 수와 오차 표시
        print("Epoch:" + str(i) + "/" + str(epoch), "Error:" + str(total_error/n_data))
        
```


![output_7_0](https://github.com/SeoMoonk/deep_learning/assets/39723465/b5b00e4e-cbbb-44c3-8c26-00e9cb965033)


   
Epoch:0/2001 Error:0.269095864634374

    
    
![output_7_2](https://github.com/SeoMoonk/deep_learning/assets/39723465/9011dd11-ad43-4e98-a41e-c16357ad429d)


Epoch:200/2001 Error:0.00943150286273521



![output_7_4](https://github.com/SeoMoonk/deep_learning/assets/39723465/d1c67907-b407-48aa-89cf-e6183d88cee9)




Epoch:400/2001 Error:0.0061120386915492495




![output_7_6](https://github.com/SeoMoonk/deep_learning/assets/39723465/15b1a622-87df-44cd-8675-a11527753369)




Epoch:600/2001 Error:0.005135635334026548




![output_7_8](https://github.com/SeoMoonk/deep_learning/assets/39723465/b081ff47-2bcb-4682-ba12-ff5cc6af7327)




Epoch:800/2001 Error:0.003958206825716445




![output_7_10](https://github.com/SeoMoonk/deep_learning/assets/39723465/d452dab1-037f-4f30-8541-975f6c11ce5c)




Epoch:1000/2001 Error:0.0032113746148401535




![output_7_12](https://github.com/SeoMoonk/deep_learning/assets/39723465/c6a2697c-cef3-48fb-bc44-0d7807c63bad)




Epoch:1200/2001 Error:0.002924562120714828


![output_7_14](https://github.com/SeoMoonk/deep_learning/assets/39723465/96418b19-3d26-4e7e-8e72-f5a03cb7fcef)





Epoch:1400/2001 Error:0.0028079611547557675



![output_7_16](https://github.com/SeoMoonk/deep_learning/assets/39723465/39695e68-09c2-45ff-8d83-bbf761113647)



Epoch:1600/2001 Error:0.003199002346202105



![output_7_18](https://github.com/SeoMoonk/deep_learning/assets/39723465/fd6ca665-c1b8-4e29-821d-0cd6fc7f615f)



Epoch:1800/2001 Error:0.0027546725091276364




![output_7_20](https://github.com/SeoMoonk/deep_learning/assets/39723465/a7974007-8ca1-4ddb-897d-fa879a1797ac)




Epoch:2000/2001 Error:0.0025620825310781275
