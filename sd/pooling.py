import numpy as np

class MaxPool():
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape

        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1

        out = np.zeros((N, C, H_out, W_out))

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        patch = x[n, c, h_start:h_end, w_start:w_end]
                        out[n, c, i, j] = np.max(patch)

        self.cache = x
        return out

    def backward(self, dout):
        x = self.cache
        N, C, H, W = x.shape

        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1

        dx = np.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        patch = x[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(patch)

                        mask = (patch == max_val)

                        dx[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, i, j]

        return dx