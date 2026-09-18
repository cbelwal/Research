"""
Code to return All_C_hat_u without normalization

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
User and session interaction data from the database

Outputs: 
All_C_hat_u : Dictionary of dictionaries to represent a sparse matrix containing non-normalized tool call frequencies for each user
"""
def Algorithm_Data_Raw():
    all_C_hat_u_1 = {} # Dictionary of dictionaries to hold C_hat_u_1 for all users
    allUsersIds = dataPrepHelper.get_all_user_ids()
    for idx in tqdm(range(0,len(allUsersIds))):
        userId = allUsersIds[idx]
        C_u = {}
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

        all_C_hat_u_1[userId] = C_u
    return all_C_hat_u_1
  