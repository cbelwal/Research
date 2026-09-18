"""
Implements a simple linear layer using PyTorch.

This a multi-layer NN

Not used in the paper, only for exploratory purposes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CMultiLayer(nn.Module):
    def __init__(self, embeddingDimensions=4):
        super().__init__()
        # Need to set seed for reproducibility, has to be defined 
        # by setting up layet as weights are assigned at that level
        torch.manual_seed(1) 
        
        embeddingDimensionsHalf = embeddingDimensions // 2
        hiddenDimensions = embeddingDimensions * 2
        self.linear_1 = nn.Linear(embeddingDimensions,
                                 hiddenDimensions, 
                                     bias=False)
        self.linear_2 = nn.Linear(hiddenDimensions,
                                  hiddenDimensions, 
                                     bias=False)
        self.linear_3 = nn.Linear(hiddenDimensions,
                                 1, 
                                 bias=False)
        

    def forward(self, x):
        y_pred = self.linear_1(x)
        y_pred = F.gelu(y_pred)
        y_pred = self.linear_2(y_pred)
        y_pred = F.gelu(y_pred)
        y_pred = self.linear_3(y_pred)
        y_pred = F.sigmoid(y_pred)
            
        return y_pred