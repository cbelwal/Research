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
from Algorithms.Alg_2_AutoEncoder import Alg_2_AutoEncoder
from Algorithms.Alg_3_MatrixFactorization import Alg_3_MatrixFactorization
from Algorithms.Alg_Baseline_PCA import Alg_Baseline_PCA
from Experiments.ExecuteExperiments.Helpers.CDistanceFunctions import CDistanceFunctions

class CTestData_Simple(IUserToolMatrix):
    def __init__(self):
        self.NumberOfUsers = 4
        self.NumberOfTools = 4

    def NumberOfUsers(self):
        return self.NumberOfUsers
    
    def NumberOfTools(self):
        return self.NumberOfTools

    """
    Returns the MAT_u_tau matrix as a tensor
    Shape: (totalNumberOfUsers, totalNumberOfTools)
    """
    def get_MAT_u_tau(self):
        MAT_tau_u = torch.zeros(self.NumberOfUsers, self.NumberOfTools)
        # ------ Manually Assign Values
        MAT_tau_u[0][0] = 1 # user 0, tool 0
        MAT_tau_u[0][1] = 2 # user 0, tool 1
        MAT_tau_u[0][2] = 2 # user 0, tool 2
        MAT_tau_u[0][3] = 0 # user 0, tool 3
        MAT_tau_u[1][0] = 3 # user 1, tool 0
        MAT_tau_u[1][1] = 1 # user 1, tool 1
        MAT_tau_u[1][2] = 0 # user 1, tool 2
        MAT_tau_u[1][3] = 2 # user 1, tool 3
        # Canary #1: user 2 has same values as user 0
        MAT_tau_u[2][0] = 1 # user 2, tool 0
        MAT_tau_u[2][1] = 2 # user 2, tool 1
        MAT_tau_u[2][2] = 2 # user 2, tool 2
        MAT_tau_u[2][3] = 0 # user 2, tool 3
        # Canary #2: user 3 has close values as user 1
        MAT_tau_u[3][0] = 3 # user 1, tool 0
        MAT_tau_u[3][1] = 1 # user 1, tool 1
        MAT_tau_u[3][2] = 1 # user 1, tool 2 -> Only difference
        MAT_tau_u[3][3] = 2 # user 1, tool 3
        #------------------------
        # Normalize the matrix values between 0 and 1 for each row
        for userId in range(self.NumberOfUsers):
            row_sum = torch.sum(MAT_tau_u[userId])
            if row_sum > 0:
                MAT_tau_u[userId] = MAT_tau_u[userId] / row_sum
        
        return MAT_tau_u
    
# For local testing only
if __name__== "__main__":
    testData = CTestData_Simple()
    (MAT_E,loss_for_user) = Alg_3_MatrixFactorization(embeddingDimensions=2,
                                                     testData=testData)
    #(MAT_E,loss_for_user) = Alg_Baseline_PCA(embeddingDimensions=2,
    #                                           testData=testData)
    '''
    In PyTorch, the .item() method is used to extract the value 
    from a single-element tensor and convert it into a standard 
    Python number (e.g., int or float). 
    '''
    print("Best losses for each user:", loss_for_user.tolist())
    
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[2],"Canary 1 - 0 and 2")
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[3],"Canary 2 - 1 and 3")
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[1],"0 and 1")
 