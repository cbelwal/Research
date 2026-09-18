import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.Plots.CPlotCommon import CPlotCommon
from Experiments.ExecuteExperiments.Helpers.CDistanceFunctions import CDistanceFunctions
from Experiments.ExecuteExperiments.Helpers.CDistanceAnalysis import CDistanceAnalysis

class CPlotDistance:
    def __init__(self, MAT_E,algID:int=None):
        self.MAT_E = MAT_E
        self.distanceAnalysis = CDistanceAnalysis(MAT_E)
        self.algID = algID

    def plot_distance_between_all_users(self, 
                                          useCosine=False,
                                          printValues=False,
                                          saveFile=False):                                                                       
        user_id_pairs = self.distanceAnalysis.get_all_user_id_pairs()
        
        distanceMeasure = ""
        if useCosine:
            distanceMeasure = "Cosine"
        else:
            distanceMeasure = "Euclidean" 
        title = f"Algorithm {self.algID}: {distanceMeasure} distance between all users"

        distances = []
        for (user_id_1, user_id_2) in user_id_pairs:
            embedding_1 = self.MAT_E[user_id_1-1]
            embedding_2 = self.MAT_E[user_id_2-1]
            if useCosine:
                distance = CDistanceFunctions.cosine_distance_tensors(embedding_1,embedding_2)
            else:
                distance = CDistanceFunctions.euclidean_distance_tensors(embedding_1,embedding_2)
            if printValues:
                print(f"User Ids: {user_id_1} and {user_id_2} :: Similarity: {distance.item()}")
            distances.append(distance.item())
        
        # Plot when all data points are available
        CPlotCommon.plot_histogram_y(distances,
                                    title = title,
                                    xlabel="Distance",
                                    ylabel="Number of user pairs",
                                    saveFile=saveFile)
        return
        
    def plot_distance_between_canary_users(self,
                                          canary_id,    
                                          useCosine=False,
                                          printValues=False,
                                          saveFile=False):                                                                       
        
        user_id_pairs = self.distanceAnalysis.get_all_canary_user_ids(canary_id=canary_id)
        
        distanceMeasure = ""
        if useCosine:
            distanceMeasure = "Cosine"
        else:
            distanceMeasure = "Euclidean" 
        title = f"Algorithm {self.algID}: {distanceMeasure} distance between canary {str(canary_id)} users"

        base_canary_user_id = user_id_pairs[0] # Use the 1st Canary user, they should all be similar
        # CAUTION: MAT_E is 0-indexed
        embedding_base = self.MAT_E[base_canary_user_id-1]
        distances = []
        for compare_user_id in user_id_pairs[1:]:
            embedding_compare = self.MAT_E[compare_user_id - 1]

            if useCosine:
                distance = CDistanceFunctions.cosine_distance_tensors(embedding_base,embedding_compare)
            else:
                distance = CDistanceFunctions.euclidean_distance_tensors(embedding_base,embedding_compare)
            if printValues:
                print(f"User Ids: {base_canary_user_id} and {compare_user_id} :: Similarity: {distance.item()}")
            distances.append(distance.item())
  
        # Plot when all data points are available
        CPlotCommon.plot_histogram_y(distances,
                                    title=title,
                                    xlabel="Distance",
                                    ylabel="Number of users",
                                    saveFile=saveFile)