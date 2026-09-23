import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.ExecuteExperiments.Helpers.CDistanceFunctions import CDistanceFunctions
from Experiments.Database.CDatabaseManager import CDatabaseManager  

class CDistanceAnalysis:
    def __init__(self, MAT_E):
        self.MAT_E = MAT_E
        self.dbManager = CDatabaseManager()
        self.canary_agents = self.dbManager.get_canary_agents()
        self.all_agent_ids = self.dbManager.get_all_agent_ids()

    def get_all_canary_agent_ids(self, canary_id):
        return self.canary_agents[canary_id]

    def print_similarity_for_all_canary_agents_in_category(self, canary_id):
        all_canary_agents = self.canary_agents[canary_id]
        if len(all_canary_agents) < 2:
            print(f"Not enough canary agents found for canary ID {canary_id}.")
            return
        base_canary_agent_id = all_canary_agents[0] # Use the 1st Canary agent, they should all be similar
        # CAUTION: MAT_E is 0-indexed
        embedding_base = self.MAT_E[base_canary_agent_id-1]
        print(f"*** Similarity measures between Canary Agents of Canary ID {canary_id}:")
        for compare_agent_id in all_canary_agents[1:]:
            embedding_compare = self.MAT_E[compare_agent_id-1]
            CDistanceFunctions.print_distance_measures_tensors(f"Ids {base_canary_agent_id} and {compare_agent_id}",
                                                                 embedding_base,embedding_compare)
    def print_similarity_between_all_canary_agents(self):
        self.print_similarity_for_all_canary_agents_in_category(canary_id=1)
        self.print_similarity_for_all_canary_agents_in_category(canary_id=2)

    
    def print_similarity_for_agent_pair(self, agent_id_1, agent_id_2):
        embedding_1 = self.MAT_E[agent_id_1]
        embedding_2 = self.MAT_E[agent_id_2]
        CDistanceFunctions.print_distance_measures_tensors(embedding_1,
                                                             embedding_2,
                                                             f"Ids {agent_id_1} and {agent_id_2}")
    def get_all_agent_id_pairs(self):
        return self.get_agent_id_pairs(self.all_agent_ids)
        
    def get_agent_id_pairs(self,given_agent_ids):
        agent_id_pairs = []
        for i in range(len(given_agent_ids)): # will start from 0
            # CAUTION: Agent IDs are 1-indexed in DB, but MAT_E is 0-indexed
            agent_id_1 = given_agent_ids[i]
            for j in range(i+1, len(given_agent_ids)):
                agent_id_2 = given_agent_ids[j]
                agent_id_pairs.append((agent_id_1, agent_id_2))
        return agent_id_pairs

    

            
