import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch import functional as F

from torch import Tensor


class RMSELoss(nn.Module):
    """Root Mean Squared Error Loss Function
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y):
        loss = torch.sqrt(self.mse(y_pred, y) + self.eps)
        return loss


class MSLELoss(nn.Module):
    """Mean Squared Logarithmic Error Loss Function
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))

class RMSLELoss(nn.Module):
    """Root Mean Squared Logarithmic Error Loss Function
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))