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
User and session interaction data from the database

Outputs: 
All_C_hat_u : Dictionary of dictionaries to represent a sparse matrix containing normalized tool call frequencies for each user
"""
def Algorithm_1_DataPreparation():
    all_C_hat_u_1 = {} # Dictionary of dictionaries to hold C_hat_u_1 for all users
    allUsersIds = dataPrepHelper.get_all_user_ids()
    for idx in tqdm(range(0,len(allUsersIds))):
        userId = allUsersIds[idx]
        C_u = {}
        C_hat_u = {}
        C_hat_u_1 = {}
        T_u = []
        sessionIdsForUser = dataPrepHelper.get_sessions_for_user(userId)
        total_tool_calls = 0
        for sessionId in sessionIdsForUser:
            toolIds = dataPrepHelper.get_tools_for_session(sessionId)
            for toolId in toolIds:
                total_tool_calls += 1
                if toolId in C_u:
                    C_u[toolId] += 1
                else:
                    C_u[toolId] = 1
                if toolId not in T_u:
                    T_u.append(toolId)
        # Normalize tool calls by number of sessions
        Sum_C_hat_u = 0    
        for toolId in T_u:
            C_hat_u[toolId] = C_u[toolId] / total_tool_calls # len(sessionIdsForUser)
            Sum_C_hat_u += C_hat_u[toolId]
        
        # Normalize so that value is between 0 and 1
        # so that way we can apply sigmoid
        # C_hat_u_1 is the normalized values on 1.0
        for toolId in T_u:
            C_hat_u_1[toolId] = C_hat_u[toolId] / Sum_C_hat_u

        all_C_hat_u_1[userId] = C_hat_u_1
    return all_C_hat_u_1
  