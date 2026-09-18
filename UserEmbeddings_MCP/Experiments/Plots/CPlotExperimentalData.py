"""
Routines to plot orig. data directly from the database
"""
import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.Database.CDatabaseManager import CDatabaseManager
from Experiments.Plots.CPlotCommon import CPlotCommon
from Experiments.CConfig import CConfig
from Experiments.ExecuteExperiments.Helpers.CResultsStore import CResultsStore
from Experiments.ExecuteExperiments.Helpers.CPCAAnalysis import CPCAAnalysis
from Experiments.ExecuteExperiments.Helpers.CClusteringAnalysis import CClusteringAnalysis
from Experiments.Plots.CPlotDistance import CPlotDistance

class CPlotExperimentalData:
    """
        1. Canary Data - Euclidean and Cosine distance between each user pair (points)
        2. Euclidean and Cosine sim. between each user pair (Histogram) 
        3. PCA plots for each user embedding
        4. User Clusters on euclidean and cosine distance
    """
    # Analyze only one A/g at a time, for memory efficiency
    def __init__(self,algID:int):
        self.dbManager = CDatabaseManager()
        resultsStore = CResultsStore(algID=algID)
        self.MAT_E = resultsStore.load_embeddings()
        self.loss_for_each_user = resultsStore.load_training_loss()
        self.plotDistance =  CPlotDistance(self.MAT_E,algID=algID)
        self.algID = algID
    
    # Plot euclidean and cosine distance between canary users
    def plot_canary_users(self):
        for canary_id in [1,2]:
            print(f"Plotting canary results for canary id: {canary_id}...")
            self.plotDistance.plot_distance_between_canary_users(canary_id=canary_id, useCosine=False, saveFile=True)
            self.plotDistance.plot_distance_between_canary_users(canary_id=canary_id, useCosine=True, saveFile=True)
    
    def plot_all_user_pairs(self):
        print(f"Plotting all user pairs...")
        self.plotDistance.plot_distance_between_all_users (useCosine=False, saveFile=True)
        self.plotDistance.plot_distance_between_all_users (useCosine=True, saveFile=True)


    def plot_training_loss(self):
        print(f"Plotting training loss...")
        CPlotCommon.plot_scatter_y(self.loss_for_each_user,
                                 title=f"Algorithm {self.algID}:Training Loss for each user",
                                 xlabel="User Id",
                                 ylabel="Training Loss",
                                 saveFile=True)

    def generate_all_plots(self):
        self.plot_canary_users()
        self.plot_all_user_pairs()
        self.plot_training_loss()
        self.plot_pca_embeddings_xy()
        self.plot_pca_embeddings_canary_xy()
        self.plot_clustering_xy_wcss()

    # Plot PCA dim 1 & PCA dim 2 vs user id
    def plot_pca_all_embeddings_y(self):
        print(f"Generating PCA plots...")
        pca=CPCAAnalysis(self.MAT_E)
        pca_embeddings_2d = pca.generate_pca_for_all(reduce_to_dim=2)
        # split embeddings into two lists for plotting
         
        print(pca_embeddings_2d)
        CPlotCommon.plot_scatter_y(pca_embeddings_2d,
                                 title=f"Algorithm {self.algID}:Embeddings PCA in 2 dimensions",
                                 xlabel="User Id",
                                 ylabel="PCA Dimension 2",
                                 saveFile=True)

    def plot_pca_embeddings_canary_xy(self):
        print(f"Generating PCA plots for canary users...")
        self.plot_pca_embeddings_xy(canary_id=1)
        self.plot_pca_embeddings_xy(canary_id=2)

    # Plot PCA dim 1 vs PCA dim 2
    def plot_pca_embeddings_xy(self, canary_id:int=None):
        print(f"Generating PCA plots...")
        pca=CPCAAnalysis(self.MAT_E)
        title = ""
        if canary_id is None:
            title=f"Algorithm {self.algID}:All Embeddings PCA"
            pca_embeddings_2d = pca.generate_pca_for_all(reduce_to_dim=2)
        else:
            title=f"Algorithm {self.algID}:Canary {str(canary_id)} Embeddings PCA"
            pca_embeddings_2d = pca.generate_pca_for_canary_users(canary_id=canary_id, reduce_to_dim=2)
        # split embeddings into two lists for plotting
        # X will contains all value in PCA dim 1
        # Y will contains all value in PCA dim 2
        X = [emb[0].item() for emb in pca_embeddings_2d]
        Y = [emb[1].item() for emb in pca_embeddings_2d]
         
        #print(pca_embeddings_2d)
        CPlotCommon.plot_scatter_xy(X,
                                    Y,
                                 title=title,
                                 xlabel="PCA Dimension 1",
                                 ylabel="PCA Dimension 2",
                                 saveFile=True)
    
    # Plot based on number of clusters returned from wcss
    def plot_clustering_xy_wcss(self):
        print(f"Generating clustering plots...")
        clustering = CClusteringAnalysis(self.MAT_E)
        print("Determining optimal number of clusters using elbow method...")
        wcss, optimal_clusters = clustering.determine_optimal_number_of_clusters_elbow()

        #-------------- Plot WCSS vs number of clusters
        CPlotCommon.plot_line_y(wcss,
                                 title=f"Algorithm {self.algID}:WCSS vs Number of clusters",
                                 xlabel="Number of clusters",
                                 ylabel="Within Cluster Sum of Squares (WCSS)",
                                 xstart=0,
                                 saveFile=True)

        print(f"Algorithm {self.algID}: Optimal number of clusters (kneed elbow): {optimal_clusters}, MAT_E shape: {tuple(self.MAT_E.shape)}")
        (cluster_labels, cluster_centroids) = clustering.generate_kmeans_clustering(num_clusters=optimal_clusters)

        # Project high-dimensional centroids to 2D via PCA fitted on all embeddings
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.MAT_E)
        pca = PCA(n_components=2)
        pca.fit(scaler.transform(self.MAT_E))
        centroids_2d = pca.transform(scaler.transform(cluster_centroids))

        X = [coord[0] for coord in centroids_2d]
        Y = [coord[1] for coord in centroids_2d]

        for i, (cx, cy) in enumerate(zip(X, Y)):
            print(f"  Centroid {i}: ({cx},{cy})")

        CPlotCommon.plot_scatter_xy(X,
                                    Y,
                                 title=f"Algorithm {self.algID}:Embedding Cluster Centroids",
                                 xlabel="X",
                                 ylabel="Y",
                                 saveFile=True)    

# Testing
if __name__ == "__main__":
    algID = 3
    plotter = CPlotExperimentalData(algID=algID)
    #plotter.plot_canary_users()
    #plotter.plot_training_loss()
    #plotter.plot_embeddings_pca_xy()
    #plotter.plot_embeddings_pca_y()
    plotter.plot_clustering_xy_wcss()
    #plotter.plot_pca_embeddings_canary_xy()