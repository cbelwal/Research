"""
Takes a sparse matrix as dictionary of dictionaries

and provides the Number of users x number of tools, 

MAT_u_tau matrix for generating embeddings
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
from Experiments.Database.CDatabaseManager import CDatabaseManager

FILL_VALUE = 1.0e-4

class CDataMain(IUserToolMatrix):
    def __init__(self, all_C_hat_u: dict):
        dbManager = CDatabaseManager()
        self.NumberOfUsers = len(all_C_hat_u)
        self.NumberOfTools = dbManager.get_number_of_tools()
        self.all_C_hat_u = all_C_hat_u

    def NumberOfUsers(self):
        return self.NumberOfUsers
    
    def NumberOfTools(self):
        return self.totalNumberOfTools

    """
    Returns the MAT_u_tau matrix as a tensor
    Shape: (totalNumberOfUsers, totalNumberOfTools)
    """
    def get_MAT_u_tau(self):
        # Create a tensor with specified value
    
        MAT_tau_u = torch.full((self.NumberOfUsers, self.NumberOfTools),FILL_VALUE)
        
        for userId in self.all_C_hat_u:
            C_hat_u = self.all_C_hat_u[userId]
            for toolId in C_hat_u: # Note tensor is 0-indexed, while db has id 1
                MAT_tau_u[userId-1][toolId-1] = C_hat_u[toolId]
        #    print(MAT_tau_u[userId-1][toolId-1])
        #print(MAT_tau_u)
        return MAT_tau_u