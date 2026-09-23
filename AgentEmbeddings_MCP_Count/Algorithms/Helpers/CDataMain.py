"""
Takes a sparse matrix as dictionary of dictionaries

and provides the Number of agents x number of tools,

MAT_a_tau matrix for generating embeddings
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

from Algorithms.Helpers.IAgentToolMatrix import IAgentToolMatrix
from Experiments.Database.CDatabaseManager import CDatabaseManager

FILL_VALUE = 1.0e-4

class CDataMain(IAgentToolMatrix):
    def __init__(self, all_C_hat_a: dict):
        dbManager = CDatabaseManager()
        self.NumberOfAgents = len(all_C_hat_a)
        self.NumberOfTools = dbManager.get_number_of_tools()
        self.all_C_hat_a = all_C_hat_a

    def NumberOfAgents(self):
        return self.NumberOfAgents
    
    def NumberOfTools(self):
        return self.totalNumberOfTools

    """
    Returns the MAT_a_tau matrix as a tensor
    Shape: (totalNumberOfAgents, totalNumberOfTools)
    """
    def get_MAT_a_tau(self):
        # Create a tensor with specified value
    
        MAT_tau_a = torch.full((self.NumberOfAgents, self.NumberOfTools),FILL_VALUE)
        
        for agentId in self.all_C_hat_a:
            C_hat_a = self.all_C_hat_a[agentId]
            for toolId in C_hat_a: # Note tensor is 0-indexed, while db has id 1
                MAT_tau_a[agentId-1][toolId-1] = C_hat_a[toolId]
        #    print(MAT_tau_a[agentId-1][toolId-1])
        #print(MAT_tau_a)
        return MAT_tau_a