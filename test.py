import torch
import numpy as np

a = np.arange(24).reshape(2, 3, 4)
print(a)
print(a.max(1, keepdims=True))
