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
from Algorithms.Helpers.IAgentToolMatrix import IAgentToolMatrix

from Experiments.ExecuteExperiments.Helpers.CDistanceFunctions import CDistanceFunctions
from Experiments.Database.CDatabaseManager import CDatabaseManager

FILL_VALUE = 1.0e-4

class CTestData_Database(IAgentToolMatrix):
    def __init__(self):
        dbManager = CDatabaseManager()
        self.all_C_hat_a = Algorithm_1_DataPreparation()
        self.NumberOfAgents = 4
        self.NumberOfTools = dbManager.get_number_of_tools()

    def NumberOfAgents(self):
        return self.NumberOfAgents
    
    def NumberOfTools(self):
        return self.NumberOfTools

    """
    Returns the MAT_a_tau matrix as a tensor
    Shape: (totalNumberOfAgents, totalNumberOfTools)
    """
    def get_MAT_a_tau(self):
          # Create a tensor with specified value
        MAT_tau_a = torch.full((self.NumberOfAgents,
                                self.NumberOfTools),
                                FILL_VALUE)
        
        for agentId in range(1, self.NumberOfAgents+1):
            C_hat_a = self.all_C_hat_a[agentId]
            for toolId in C_hat_a: # Note tensor is 0-indexed, while db has id 1
                MAT_tau_a[agentId-1][toolId-1] = C_hat_a[toolId]

        return MAT_tau_a

    
# For local testing only
if __name__== "__main__":
    testData = CTestData_Database()
    (MAT_E,loss_for_agent) = Alg_3_MatrixFactorization(embeddingDimensions=2,
                                                     testData=testData)
    
    print("Best losses for each agent:", loss_for_agent.tolist())
    
    MAT_tau_a = testData.get_MAT_a_tau()
    print(MAT_tau_a[0],MAT_tau_a[1])
    CDistanceFunctions.print_distance_measures_tensors("Raw Data 0 and 1",MAT_tau_a[0],MAT_tau_a[1])
    
    
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[1],"AgentIds: 1 and 2")
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[2],"AgentIds: 1 and 3")
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[3],"AgentIds: 1 and 3")
    