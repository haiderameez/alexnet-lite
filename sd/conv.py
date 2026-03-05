import numpy as np

class Conv():
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        self.C = in_channels
        self.F = out_channels
        self.K = kernel_size
        self.stride = stride
        self.pad = pad

        self.W = 0.01 * np.random.randn(self.F, self.C, self.K, self.K)
        self.b = np.zeros(self.F)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape

        H_out = (H + 2*self.pad - self.K)//self.stride + 1
        W_out = (W + 2*self.pad - self.K)//self.stride + 1

        x_padded = np.pad(x, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)))

        out = np.zeros((N, self.F, H_out, W_out))

        for n in range(N):
            for f in range(self.F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i*self.stride
                        h_end = h_start + self.K
                        w_start = j*self.stride
                        w_end = w_start + self.K

                        x_patch = x_padded[n, :, h_start:h_end, w_start:w_end]

                        out[n,f,i,j] = np.sum(x_patch * self.W[f]) + self.b[f]

        self.cache = (x, x_padded)

        return out

    def backward(self, dout):
        x, x_padded = self.cache
        N, C, H, W = x.shape

        _, _, H_out, W_out = dout.shape

        dx_padded = np.zeros_like(x_padded)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        for n in range(N):
            for f in range(self.F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i*self.stride
                        h_end = h_start + self.K
                        w_start = j*self.stride
                        w_end = w_start + self.K

                        x_patch = x_padded[n, :, h_start:h_end, w_start:w_end]

                        self.dW[f] += x_patch * dout[n,f,i,j]

                        dx_padded[n,:,h_start:h_end,w_start:w_end] += self.W[f] * dout[n,f,i,j]

        self.db = np.sum(dout, axis=(0,2,3))

        if self.pad > 0:
            dx = dx_padded[:,:,self.pad:-self.pad,self.pad:-self.pad]
        else:
            dx = dx_padded

        return dx