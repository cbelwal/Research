"""
Loads limited test data from the database
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
from Algorithms.Alg_1_DataPreparation import Algorithm_1_DataPreparation
from Algorithms.Alg_2_AutoEncoder import Alg_2_AutoEncoder
from Algorithms.Alg_3_MatrixFactorization import Alg_3_MatrixFactorization
from Algorithms.Helpers.IUserToolMatrix import IUserToolMatrix  

from Experiments.ExecuteExperiments.Helpers.CDistanceFunctions import CDistanceFunctions
from Experiments.Database.CDatabaseManager import CDatabaseManager

FILL_VALUE = 1.0e-4

class CTestData_Database(IUserToolMatrix):
    def __init__(self):
        dbManager = CDatabaseManager()
        self.all_C_hat_u = Algorithm_1_DataPreparation()
        self.NumberOfUsers = 4
        self.NumberOfTools = dbManager.get_number_of_tools()

    def NumberOfUsers(self):
        return self.NumberOfUsers
    
    def NumberOfTools(self):
        return self.NumberOfTools

    """
    Returns the MAT_u_tau matrix as a tensor
    Shape: (totalNumberOfUsers, totalNumberOfTools)
    """
    def get_MAT_u_tau(self):
          # Create a tensor with specified value
        MAT_tau_u = torch.full((self.NumberOfUsers, 
                                self.NumberOfTools),
                                FILL_VALUE)
        
        for userId in range(1, self.NumberOfUsers+1):
            C_hat_u = self.all_C_hat_u[userId]
            for toolId in C_hat_u: # Note tensor is 0-indexed, while db has id 1
                MAT_tau_u[userId-1][toolId-1] = C_hat_u[toolId]

        return MAT_tau_u

    
# For local testing only
if __name__== "__main__":
    testData = CTestData_Database()
    (MAT_E,loss_for_user) = Alg_3_MatrixFactorization(embeddingDimensions=2,
                                                     testData=testData)
    
    print("Best losses for each user:", loss_for_user.tolist())
    
    MAT_tau_u = testData.get_MAT_u_tau()
    print(MAT_tau_u[0],MAT_tau_u[1])
    CDistanceFunctions.print_distance_measures_tensors("Raw Data 0 and 1",MAT_tau_u[0],MAT_tau_u[1])
    
    
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[1],"UserIds: 1 and 2")
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[2],"UserIds: 1 and 3") 
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[3],"UserIds: 1 and 3") 
    