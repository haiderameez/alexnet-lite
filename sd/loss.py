import numpy as np

class Loss():
    def __init__(self):
        self.cache = None

    def forward(self, logits, labels):
        shifted_logits = logits - np.max(logits, axis = 1, keepdims=True)

        softmax = np.exp(shifted_logits)/np.sum(np.exp(shifted_logits), axis = 1, keepdims=True)

        SCE = -np.log(softmax[np.arange(logits.shape[0]), labels])

        loss = np.mean(SCE)

        self.cache = (softmax, labels, logits.shape[0])

        return loss
    
    def backward(self):
        softmax, labels, N = self.cache

        dlogits = softmax.copy()
        dlogits[np.arange(N), labels] -= 1
        dlogits /= N

        return dlogits