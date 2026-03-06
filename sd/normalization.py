import numpy as np

class LRN():
    def __init__(self, n=5, k=2, alpha=1e-4, beta=0.75):
        self.n = n
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape

        sq = x ** 2
        scale = np.zeros_like(x)

        half = self.n // 2

        for c in range(C):
            start = max(0, c - half)
            end = min(C, c + half + 1)

            scale[:, c, :, :] = np.sum(sq[:, start:end, :, :], axis=1)

        scale = self.k + (self.alpha / self.n) * scale
        out = x / (scale ** self.beta)

        self.cache = (x, scale)

        return out

    def backward(self, dout):
        x, scale = self.cache

        N, C, H, W = x.shape
        dx = np.zeros_like(x)

        half = self.n // 2

        for c in range(C):
            start = max(0, c - half)
            end = min(C, c + half + 1)

            dx[:, c, :, :] += dout[:, c, :, :] / (scale[:, c, :, :] ** self.beta)

            for j in range(start, end):
                dx[:, c, :, :] -= (
                    (2 * self.alpha * self.beta / self.n)
                    * x[:, c, :, :]
                    * x[:, j, :, :]
                    * dout[:, j, :, :]
                    / (scale[:, j, :, :] ** (self.beta + 1))
                )

        return dx