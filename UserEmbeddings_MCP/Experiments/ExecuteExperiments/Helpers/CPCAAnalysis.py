"""
Will take the embeddings generated from various algorithms and perform PCA analysis on them
to visualize the user embeddings in 2D space.
"""
import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.Database.CDatabaseManager import CDatabaseManager 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

class CPCAAnalysis:
    def __init__(self, MAT_E):
        self.MAT_E = MAT_E
        self.dbManager = CDatabaseManager()
        self.canary_users = self.dbManager.get_canary_users()
    
    def generate_pca_for_canary_users(self, canary_id=1, reduce_to_dim:int=2):
        canary_user_indices = [user_id - 1 for user_id in self.canary_users[canary_id]] # Convert to 0-indexed
        canary_embeddings = self.MAT_E[canary_user_indices]
        pca_embeddings = self.generate_pca_for_given_values(canary_embeddings, reduce_to_dim=reduce_to_dim)
        return pca_embeddings

    def generate_pca_for_all(self,reduce_to_dim:int=2):
        pca_embeddings = self.generate_pca_for_given_values(self.MAT_E, reduce_to_dim=reduce_to_dim) # CAUTION: MAT_E is 0-indexed
        return pca_embeddings

    def generate_pca_for_given_values(self, embeddings, reduce_to_dim:int=2):
        # Convert embeddings dict to a 2D array for PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embeddings)
        model = PCA(n_components=reduce_to_dim) # Retain 2 principal components
        # Fit PCA to the scaled data and transform it
        pca_components = model.fit_transform(X_scaled)
        return pca_components

# For testing purposes
if __name__ == "__main__":
    MAT_E = torch.zeros(2, 3) # 2 users x 3 tools
    # ------ Manually Assign Values
    MAT_E[0][0] = 1 # user 0, tool 0
    MAT_E[0][1] = 2 # user 0, tool 1
    MAT_E[0][2] = 2 # user 0, tool 2
    MAT_E[1][0] = 3 # user 1, tool 0
    MAT_E[1][1] = 1 # user 1, tool 1
    MAT_E[1][2] = 0 # user 1, tool 2
    
    pcaAnalyzer = CPCAAnalysis(MAT_E)
    pca_embeddings = pcaAnalyzer.generate_pca_for_all()   
    print(pca_embeddings)            
