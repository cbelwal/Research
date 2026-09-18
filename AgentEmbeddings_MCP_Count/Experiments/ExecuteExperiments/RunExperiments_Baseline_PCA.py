# These compute the baselines for PCA on synthetic data

import os,sys
import datetime
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------

from Algorithms.Alg_Data_Raw import Algorithm_Data_Raw
from Algorithms.Alg_Baseline_PCA import Alg_Baseline_PCA

from Algorithms.Helpers.CDataMain import CDataMain
from Experiments.CConfig import CConfig
from Experiments.ExecuteExperiments.Helpers.CResultsStore import CResultsStore
from Experiments.ExecuteExperiments.Helpers.CDistanceAnalysis import CDistanceAnalysis

PCA_ALG_ID = 11

def run_pca_on_synthetic_data(All_C_hat_u:dict):     
    testData = CDataMain(All_C_hat_u)
    embeddingDimensions = CConfig.EMBEDDING_DIMENSIONS
   
    (MAT_E, loss_for_each_user) = Alg_Baseline_PCA(
                embeddingDimensions=embeddingDimensions,
                testData=testData)
    print(f"Generated Embeddings with PCA") 
    return (MAT_E, loss_for_each_user)


def store_results_in_file(MAT_E, loss_for_each_user, algID:int):
    store = CResultsStore(algID)  
    # Store embeddings in file
    store.store_embeddings(MAT_E)
    print(f"Agent embeddings stored in file")
    # Store training loss in file
    store.store_training_loss(loss_for_each_user)
    print(f"Training loss stored in file")


# Run the experiments and store embeddings and losses in file
if __name__== "__main__":
    print(f"Starting baseline experiments on synthetic data at time {datetime.datetime.now()}...")
    print("Preparing raw data...")
    # Step 1: Data Preparation
    All_C_hat_u = Algorithm_Data_Raw()

    # Compute PCA embeddings
    print(f"Computing embeddings for PCA...")
    (MAT_E, loss_for_each_user) = \
        run_pca_on_synthetic_data(All_C_hat_u)
    print(f"Storing embeddings for PCA...")
    store_results_in_file(MAT_E, loss_for_each_user,PCA_ALG_ID)
    print(f"Experiments completed at time {datetime.datetime.now()}.")

       
 