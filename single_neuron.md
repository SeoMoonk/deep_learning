```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
X1 = np.arange(-1.0, 1.0, 0.2)
X2 = np.arange(-1.0, 1.0, 0.2)
```


```python
X1
```




    array([-1.00000000e+00, -8.00000000e-01, -6.00000000e-01, -4.00000000e-01,
           -2.00000000e-01, -2.22044605e-16,  2.00000000e-01,  4.00000000e-01,
            6.00000000e-01,  8.00000000e-01])




```python
X2
```




    array([-1.00000000e+00, -8.00000000e-01, -6.00000000e-01, -4.00000000e-01,
           -2.00000000e-01, -2.22044605e-16,  2.00000000e-01,  4.00000000e-01,
            6.00000000e-01,  8.00000000e-01])




```python
Z = np.zeros((10,10))
```


```python
W = np.array([2.5, 3.0])
bias = np.array([0.1])
```


```python
for i in range(10):
    for j in range(10):
        X = np.array( [ X1[i], X2[j] ] )
        u = np.dot(X, W.T) + bias   #T => 전치행렬을 만들어줌.
        y = 1 / (1 + np.exp(-u))
        Z[j][i] = y[0]
```


```python
plt.imshow(Z, 'gray', vmin=0.0, vmax=1.0)
plt.colorbar()
plt.show()
```

![output_7_0](https://github.com/SeoMoonk/deep_learning/assets/39723465/9f73c9f0-a962-45c2-b081-7b1636f1a8e1)
