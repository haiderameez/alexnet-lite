import numpy as np

class Dropout():
    def __init__(self, p):
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            out = x * self.mask
        else:
            self.mask = None
            out = x
        return out
    
    def backward(self, dout):
        if self.mask is None:
            return dout
        dx = dout * self.mask
        return dx