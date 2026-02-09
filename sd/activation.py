import numpy as np

class ReLU():
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        x = self.cache
        dx = dout*(x > 0)
        return dx