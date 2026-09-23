"""
Code to return All_C_hat_a without normalization

These non-normalized tool calls are used in PCA, raw distances etc.

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
All_C_hat_a : Dictionary of dictionaries to represent a sparse matrix containing non-normalized tool call frequencies for each agent
"""
def Algorithm_Data_Raw():
    all_C_hat_a_1 = {} # Dictionary of dictionaries to hold C_hat_a_1 for all agents
    allAgentsIds = dataPrepHelper.get_all_agent_ids()
    for idx in tqdm(range(0,len(allAgentsIds))):
        agentId = allAgentsIds[idx]
        C_a = {}
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

        all_C_hat_a_1[agentId] = C_a
    return all_C_hat_a_1
  