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

from Algorithms.Helpers.IAgentToolMatrix import IAgentToolMatrix
from Experiments.ExecuteExperiments.Helpers.CDistanceFunctions import CDistanceFunctions
from Algorithms.Alg_2_AutoEncoder import Alg_2_AutoEncoder
from Algorithms.Alg_3_MatrixFactorization import Alg_3_MatrixFactorization

FILL_VALUE = 1.0e-4

class CTestData_Tensor(IAgentToolMatrix):
    def __init__(self):
        self.NumberOfAgents = 2
        self.NumberOfTools = 3000

    def NumberOfAgents(self):
        return self.NumberOfAgents
    
    def NumberOfTools(self):
        return self.NumberOfTools

    """
    Returns the MAT_a_tau matrix as a tensor
    Shape: (totalNumberOfAgents, totalNumberOfTools)
    """
    def get_MAT_a_tau(self):
        MAT_tau_a = torch.full((self.NumberOfAgents, self.NumberOfTools),FILL_VALUE)
        
        # Manually Assign Values
        # Agent 0 ----
        MAT_tau_a[0][2] = 0.053
        MAT_tau_a[0][5] = 0.063

        # Agent 1 ----
        MAT_tau_a[1][800] = 0.075
        MAT_tau_a[0][5] = 0.04

        return MAT_tau_a

    
# For local testing only
if __name__== "__main__":
    testData = CTestData_Tensor()
    (MAT_E,loss_for_agent) = Alg_3_MatrixFactorization(embeddingDimensions=2,
                                                     testData=testData)
    '''
    In PyTorch, the .item() method is used to extract the value 
    from a single-element tensor and convert it into a standard 
    Python number (e.g., int or float). 
    '''
    print("Best losses for each agent:", loss_for_agent.tolist())
    
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[1],"0 and 1") 
    
   