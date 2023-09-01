```python
# sigmoid
import numpy as np
import matplotlib.pyplot as plt 

def sigmoid_function(x):
    return 1/(1+np.exp(-x)) 

x = np.linspace(-5, 5)
y = sigmoid_function(x) 

plt.plot(x, y)
plt.show()
```

![output_0_0](https://github.com/SeoMoonk/deep_learning/assets/39723465/31242d4c-4258-48f2-876e-2c39a3536b73)




```python
# softmax
def softmax_function(x):
    return np.exp(x)/np.sum(np.exp(x)) 

y = softmax_function(np.array([1,2,3]))
print(y)
```

    [0.09003057 0.24472847 0.66524096]
    

