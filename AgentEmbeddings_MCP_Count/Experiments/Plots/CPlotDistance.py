import os,sys
import torch
import torch.nn.functional as F
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
from Experiments.CConfig import CConfig

class CPlotDistance:
    def __init__(self, MAT_E,algID:int=None):
        self.MAT_E = MAT_E
        self.distanceAnalysis = CDistanceAnalysis(MAT_E)
        self.algID = algID

    def plot_distance_between_all_agents(self,
                                          useCosine=False,
                                          printValues=False,
                                          saveFile=False):                                                                       
        distanceMeasure = ""
        if useCosine:
            distanceMeasure = "Cosine"
        else:
            distanceMeasure = "Euclidean" 
        title = f"Algorithm {self.algID}: {distanceMeasure} distance between all agents"

        number_of_agents = self.MAT_E.shape[0]
        total_pairs = number_of_agents * (number_of_agents - 1) // 2
        sample_count = min(CConfig.MAX_PAIR_SAMPLES, total_pairs)
        generator = torch.Generator().manual_seed(42)
        left_indices = torch.randint(
            0,
            number_of_agents,
            (sample_count,),
            generator=generator,
        )
        right_indices = torch.randint(
            0,
            number_of_agents - 1,
            (sample_count,),
            generator=generator,
        )
        right_indices += (right_indices >= left_indices).long()

        left = self.MAT_E[left_indices]
        right = self.MAT_E[right_indices]
        if useCosine:
            distances = (
                1.0 - F.cosine_similarity(left, right, dim=1)
            ).cpu().numpy()
        else:
            distances = torch.linalg.vector_norm(
                left - right,
                dim=1,
            ).cpu().numpy()
        if printValues:
            print(distances)
        
        # Plot when all data points are available
        CPlotCommon.plot_histogram_y(distances,
                                    title = title,
                                    xlabel="Distance",
                                    ylabel=f"Sampled agent pairs (n={sample_count})",
                                    saveFile=saveFile)
        return
        
    def plot_distance_between_canary_agents(self,
                                          canary_id,    
                                          useCosine=False,
                                          printValues=False,
                                          saveFile=False):                                                                       
        
        agent_id_pairs = self.distanceAnalysis.get_all_canary_agent_ids(canary_id=canary_id)
        
        distanceMeasure = ""
        if useCosine:
            distanceMeasure = "Cosine"
        else:
            distanceMeasure = "Euclidean" 
        title = f"Algorithm {self.algID}: {distanceMeasure} distance between canary {str(canary_id)} agents"

        base_canary_agent_id = agent_id_pairs[0] # Use the 1st Canary agent, they should all be similar
        # CAUTION: MAT_E is 0-indexed
        embedding_base = self.MAT_E[base_canary_agent_id-1]
        comparison_embeddings = self.MAT_E[
            [agent_id - 1 for agent_id in agent_id_pairs[1:]]
        ]
        base_embeddings = embedding_base.expand_as(comparison_embeddings)
        if useCosine:
            distances = (
                1.0
                - F.cosine_similarity(
                    base_embeddings,
                    comparison_embeddings,
                    dim=1,
                )
            ).cpu().numpy()
        else:
            distances = torch.linalg.vector_norm(
                base_embeddings - comparison_embeddings,
                dim=1,
            ).cpu().numpy()
        if printValues:
            print(distances)
  
        # Plot when all data points are available
        CPlotCommon.plot_histogram_y(distances,
                                    title=title,
                                    xlabel="Distance",
                                    ylabel="Number of agents",
                                    saveFile=saveFile)