import numpy as np

class FC():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = 0.01*np.random.randn(input_dim, output_dim)
        self.b = np.zeros(output_dim)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.cache = None

    def forward(self, x):
        self.cache = x
        
        if x.shape[-1] != self.input_dim:
            raise ValueError("Incorrect shape")
        
        out = x@self.W + self.b
        return out
    
    def backward(self, dout):
        x = self.cache
        
        self.dW = x.T@dout
        self.db = np.sum(dout, axis = 0)

        dx = dout@self.W.T

        return dx