"""
This applies a polynomail regression model
Do note that with x = 1, it will not be that effective 
and is primary meant to compare against the NN Model 

THIS IS NOT USED IN THE PAPER
"""
import torch
from torch import nn

class CPolynomialRegression(nn.Module):
    # A polynomial regression of degree d has d+1 parameters
    def __init__(self, embeddingDimensions:int):
        super().__init__()
        self.degree = embeddingDimensions - 1
        # Polynomial of degree d is of type:
        # y = c0 + c1*x^1 + c2*x^2 + ... + cd*x^d
        # here c0 = c0 * x^0
        # Hence we need (degree + 1) coefficients
        # and will not use the bias term as it will add one more coefficient

        self.linear = nn.Linear(self.degree+1,
                                1, 
                                bias=False) 
         

    def forward(self, x):
        # power= 0,1,2,...,degree
        powers = torch.arange(0, self.degree+1, 
                              device=x.device, 
                              dtype=torch.float32).view(1, -1)
        #y_pred = self.coeffs[0] # Constant term
     
        x_poly = x.pow(powers)
        # The linear layer will take care of multiplying 
        # with coeffs and summing them up
        # Or it will do the following:
        # for i in range(1, self.degree + 1):
        #    y_pred += weights[i] * (x ** i)
        # weights[i] is the coefficient
        y_pred = self.linear(x_poly)
        return y_pred