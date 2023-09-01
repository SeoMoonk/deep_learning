
```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
X = np.arange( -1.0, 1.0, 0.1 )
Y = np.arange( -1.0, 1.0, 0.1 )
```


```python
w_im = np.array( [ [ 1.0, 2.0 ], [ 2.0, 3.0 ] ] )
w_mo = np.array( [ [ -1.0, 1.0 ], [ 1.0, -1.0 ] ])

b_im = np.array( [ 0.3, -0.3 ] )
b_mo = np.array( [ 0.4, 0.1 ] )
```


```python
def middle_layer(x, w, b): #은닉층 (시그모이드)
    u = np.dot(x, w) + b
    return 1 / (1 + np.exp(-u))  #sigmoid
```


```python
def output_layer(x, w, b): #출력층 (소프트맥스)
    u = np.dot(x, w) + b
    return np.exp(u) / np.sum(np.exp(u))
```


```python
x_1 = []
y_1 = []

x_2 = []
y_2 = []
```


```python
for i in range(20):
    for j in range(20):
        inp = np.array( [X[i], Y[j] ])
        mid = middle_layer(inp, w_im, b_im)
        out = output_layer(mid, w_mo, b_mo)
        
        if out[0] > out[1]:
            x_1.append(X[i])
            y_1.append(Y[j])
        else:
            x_2.append(X[i])
            y_2.append(Y[j])
```


```python
plt.scatter(x_1, y_1, marker='+')
plt.scatter(x_2, y_2, marker='o')
plt.show()
```
    

![output_7_0](https://github.com/SeoMoonk/deep_learning/assets/39723465/19d907d2-3889-4e1a-bc10-f626f1b73c5c)

