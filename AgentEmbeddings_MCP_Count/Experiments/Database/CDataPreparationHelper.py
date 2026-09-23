import os,sys
# ----------------------------------------------
# Ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------

from Experiments.CConfig import CConfig
from Experiments.Database.CDatabaseManager import CDatabaseManager


class CDataPreparationHelper:
   def __init__(self):
      self.dbManager = CDatabaseManager()
      self.all_sessions_data_cache = self.dbManager.get_all_tools_and_sessions()

   def get_all_agent_ids(self):
      return self.dbManager.get_all_agent_ids()

   def get_sessions_for_agent(self,agentId):
      return self.dbManager.get_sessions_for_agent(agentId)

    # Get tools for a session from cache for faster reads 
   def get_tools_for_session(self, sessionId):
       return self.all_sessions_data_cache[sessionId]
       #return self.dbManager.get_tools_for_session(sessionId)
   
    