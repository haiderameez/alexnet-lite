import torch
from torch import nn
import math

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0)