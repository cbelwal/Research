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
        self.canary_users = self.dbManager.get_canary_users()
        self.all_user_ids = self.dbManager.get_all_user_ids()

    def get_all_canary_user_ids(self, canary_id):
        return self.canary_users[canary_id]

    def print_similarity_for_all_canary_users_in_category(self, canary_id):
        all_canary_users = self.canary_users[canary_id]
        if len(all_canary_users) < 2:
            print(f"Not enough canary users found for canary ID {canary_id}.")
            return
        base_canary_user_id = all_canary_users[0] # Use the 1st Canary user, they should all be similar
        # CAUTION: MAT_E is 0-indexed
        embedding_base = self.MAT_E[base_canary_user_id-1]
        print(f"*** Similarity measures between Canary Users of Canary ID {canary_id}:")
        for compare_user_id in all_canary_users[1:]:
            embedding_compare = self.MAT_E[compare_user_id-1]
            CDistanceFunctions.print_distance_measures_tensors(f"Ids {base_canary_user_id} and {compare_user_id}", 
                                                                 embedding_base,embedding_compare)
    def print_similarity_between_all_canary_users(self):
        self.print_similarity_for_all_canary_users_in_category(canary_id=1)
        self.print_similarity_for_all_canary_users_in_category(canary_id=2)

    
    def print_similarity_for_user_pair(self, user_id_1, user_id_2):
        embedding_1 = self.MAT_E[user_id_1]
        embedding_2 = self.MAT_E[user_id_2]
        CDistanceFunctions.print_distance_measures_tensors(embedding_1,
                                                             embedding_2,
                                                             f"Ids {user_id_1} and {user_id_2}") 
    def get_all_user_id_pairs(self):
        return self.get_user_id_pairs(self.all_user_ids)
        
    def get_user_id_pairs(self,given_user_ids):
        user_id_pairs = []
        for i in range(len(given_user_ids)): # will start from 0
            # CAUTION: User IDs are 1-indexed in DB, but MAT_E is 0-indexed
            user_id_1 = given_user_ids[i]
            for j in range(i+1, len(given_user_ids)):
                user_id_2 = given_user_ids[j]
                user_id_pairs.append((user_id_1, user_id_2))
        return user_id_pairs

    

            
