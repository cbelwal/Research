"""
Contains hard coded static test data for
unit tests
"""
import torch

import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------

from Algorithms.Helpers.IUserToolMatrix import IUserToolMatrix  
from Experiments.ExecuteExperiments.Helpers.CDistanceFunctions import CDistanceFunctions
from Algorithms.Alg_2_AutoEncoder import Alg_2_AutoEncoder
from Algorithms.Alg_3_MatrixFactorization import Alg_3_MatrixFactorization

FILL_VALUE = 1.0e-4

class CTestData_Tensor(IUserToolMatrix):
    def __init__(self):
        self.NumberOfUsers = 2
        self.NumberOfTools = 3000

    def NumberOfUsers(self):
        return self.NumberOfUsers
    
    def NumberOfTools(self):
        return self.NumberOfTools

    """
    Returns the MAT_u_tau matrix as a tensor
    Shape: (totalNumberOfUsers, totalNumberOfTools)
    """
    def get_MAT_u_tau(self):
        MAT_tau_u = torch.full((self.NumberOfUsers, self.NumberOfTools),FILL_VALUE)
        
        # Manually Assign Values
        # User 0 ----
        MAT_tau_u[0][2] = 0.053
        MAT_tau_u[0][5] = 0.063

        # User 1 ----
        MAT_tau_u[1][800] = 0.075
        MAT_tau_u[0][5] = 0.04

        return MAT_tau_u

    
# For local testing only
if __name__== "__main__":
    testData = CTestData_Tensor()
    (MAT_E,loss_for_user) = Alg_3_MatrixFactorization(embeddingDimensions=2,
                                                     testData=testData)
    '''
    In PyTorch, the .item() method is used to extract the value 
    from a single-element tensor and convert it into a standard 
    Python number (e.g., int or float). 
    '''
    print("Best losses for each user:", loss_for_user.tolist())
    
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[1],"0 and 1") 
    
   