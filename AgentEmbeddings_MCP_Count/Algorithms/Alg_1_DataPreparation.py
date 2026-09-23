"""
Code to run Algorithm 1 (Data Preparation) given in paper

Variables names have been kept similar to those in the paper for ease of understanding.

The algorithms has dependencies on other Classes which are defined in different files.
"""
import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------
from tqdm import tqdm
from Experiments.Database.CDataPreparationHelper import CDataPreparationHelper

dataPrepHelper = CDataPreparationHelper() # This will take time due to cache creation

"""
Inputs: 
Agent and session interaction data from the database

Outputs: 
All_C_hat_a : Dictionary of dictionaries to represent a sparse matrix containing normalized tool call frequencies for each agent
"""
def Algorithm_1_DataPreparation():
    all_C_hat_a_1 = {} # Dictionary of dictionaries to hold C_hat_a_1 for all agents
    allAgentsIds = dataPrepHelper.get_all_agent_ids()
    for idx in tqdm(range(0,len(allAgentsIds))):
        agentId = allAgentsIds[idx]
        C_a = {}
        C_hat_a = {}
        C_hat_a_1 = {}
        T_a = []
        sessionIdsForAgent = dataPrepHelper.get_sessions_for_agent(agentId)
        total_tool_calls = 0
        for sessionId in sessionIdsForAgent:
            toolIds = dataPrepHelper.get_tools_for_session(sessionId)
            for toolId in toolIds:
                total_tool_calls += 1
                if toolId in C_a:
                    C_a[toolId] += 1
                else:
                    C_a[toolId] = 1
                if toolId not in T_a:
                    T_a.append(toolId)
        # Normalize tool calls by number of sessions
        Sum_C_hat_a = 0
        for toolId in T_a:
            C_hat_a[toolId] = C_a[toolId] / total_tool_calls # len(sessionIdsForAgent)
            Sum_C_hat_a += C_hat_a[toolId]
        
        # Normalize so that value is between 0 and 1
        # so that way we can apply sigmoid
        # C_hat_a_1 is the normalized values on 1.0
        for toolId in T_a:
            C_hat_a_1[toolId] = C_hat_a[toolId] / Sum_C_hat_a

        all_C_hat_a_1[agentId] = C_hat_a_1
    return all_C_hat_a_1
  