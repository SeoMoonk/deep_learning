```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
X = np.arange(-1.0, 1.0, 0.2)
Y = np.arange(-1.0, 1.0, 0.2)

Z = np.zeros((10, 10))
```


```python
w_im = np.array( [ [ 4.0, 4.0 ],
                   [ 4.0, 4.0 ]  ] )#input 에서 middle로 가는 가중치
w_mo = np.array( [ [ 1.0 ],
                   [ -1.0 ] ] ) #row는 1개, column은 2개 => (출력이 하나)

b_im = np.array([3.0, -3.0])
b_mo = np.array([0.1])
```


```python
def middle_layer(x, w, b): #은닉층
    u = np.dot(x, w) + b
    return 1 / (1 + np.exp(-u))  #sigmoid
```


```python
def output_layer(x, w, b):
    u = np.dot(x, w) + b
    return u;
```


```python
for i in range(10):
    for j in range(10):
        inp = np.array( [X[i], Y[j] ])
        mid = middle_layer(inp, w_im, b_im)
        out = output_layer(mid, w_mo, b_mo)
        
        Z[j][i] = out[0]
```


```python
plt.imshow(Z, 'gray', vmin=0.0, vmax=1.0)
plt.colorbar()
plt.show()
```


    ![output_6_0](https://github.com/SeoMoonk/deep_learning/assets/39723465/00a668e2-c1a8-4aee-86ec-d69975554dd1)
    



```python

```
