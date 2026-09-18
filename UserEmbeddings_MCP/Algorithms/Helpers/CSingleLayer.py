"""
Implements a simple linear layer using PyTorch.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CSingleLayer(nn.Module):
    def __init__(self, embeddingDimensions=4):
        super().__init__()
        # Need to set seed for reproducibility, has to be defined 
        # by setting up layet as weights are assigned at that level
        torch.manual_seed(1) 
        # nn.Linear (input features, output features)
        self.linear = nn.Linear(embeddingDimensions,
                                1, 
                                bias=False)

    def forward(self, x):
        y_pred = self.linear(x)
        y_pred = F.sigmoid(y_pred)
        return y_pred