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
from Algorithms.Alg_2_AutoEncoder import Alg_2_AutoEncoder
from Algorithms.Alg_3_MatrixFactorization import Alg_3_MatrixFactorization
from Algorithms.Alg_Baseline_PCA import Alg_Baseline_PCA
from Experiments.ExecuteExperiments.Helpers.CDistanceFunctions import CDistanceFunctions

class CTestData_Simple(IAgentToolMatrix):
    def __init__(self):
        self.NumberOfAgents = 4
        self.NumberOfTools = 4

    def NumberOfAgents(self):
        return self.NumberOfAgents
    
    def NumberOfTools(self):
        return self.NumberOfTools

    """
    Returns the MAT_a_tau matrix as a tensor
    Shape: (totalNumberOfAgents, totalNumberOfTools)
    """
    def get_MAT_a_tau(self):
        MAT_tau_a = torch.zeros(self.NumberOfAgents, self.NumberOfTools)
        # ------ Manually Assign Values
        MAT_tau_a[0][0] = 1 # agent 0, tool 0
        MAT_tau_a[0][1] = 2 # agent 0, tool 1
        MAT_tau_a[0][2] = 2 # agent 0, tool 2
        MAT_tau_a[0][3] = 0 # agent 0, tool 3
        MAT_tau_a[1][0] = 3 # agent 1, tool 0
        MAT_tau_a[1][1] = 1 # agent 1, tool 1
        MAT_tau_a[1][2] = 0 # agent 1, tool 2
        MAT_tau_a[1][3] = 2 # agent 1, tool 3
        # Canary #1: agent 2 has same values as agent 0
        MAT_tau_a[2][0] = 1 # agent 2, tool 0
        MAT_tau_a[2][1] = 2 # agent 2, tool 1
        MAT_tau_a[2][2] = 2 # agent 2, tool 2
        MAT_tau_a[2][3] = 0 # agent 2, tool 3
        # Canary #2: agent 3 has close values as agent 1
        MAT_tau_a[3][0] = 3 # agent 1, tool 0
        MAT_tau_a[3][1] = 1 # agent 1, tool 1
        MAT_tau_a[3][2] = 1 # agent 1, tool 2 -> Only difference
        MAT_tau_a[3][3] = 2 # agent 1, tool 3
        #------------------------
        # Normalize the matrix values between 0 and 1 for each row
        for agentId in range(self.NumberOfAgents):
            row_sum = torch.sum(MAT_tau_a[agentId])
            if row_sum > 0:
                MAT_tau_a[agentId] = MAT_tau_a[agentId] / row_sum
        
        return MAT_tau_a
    
# For local testing only
if __name__== "__main__":
    testData = CTestData_Simple()
    (MAT_E,loss_for_agent) = Alg_3_MatrixFactorization(embeddingDimensions=2,
                                                     testData=testData)
    #(MAT_E,loss_for_agent) = Alg_Baseline_PCA(embeddingDimensions=2,
    #                                           testData=testData)
    '''
    In PyTorch, the .item() method is used to extract the value 
    from a single-element tensor and convert it into a standard 
    Python number (e.g., int or float). 
    '''
    print("Best losses for each agent:", loss_for_agent.tolist())
    
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[2],"Canary 1 - 0 and 2")
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[3],"Canary 2 - 1 and 3")
    CDistanceFunctions.print_distance_measures_tensors(MAT_E[0],MAT_E[1],"0 and 1")
 