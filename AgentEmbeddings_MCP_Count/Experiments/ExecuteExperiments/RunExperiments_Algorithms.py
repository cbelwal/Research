import os,sys
import datetime
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------

from Algorithms.Alg_1_DataPreparation import Algorithm_1_DataPreparation
from Algorithms.Alg_2_AutoEncoder import Alg_2_AutoEncoder
from Algorithms.Alg_3_MatrixFactorization import Alg_3_MatrixFactorization
from Algorithms.Alg_Baseline_PCA import Alg_Baseline_PCA

from Algorithms.Helpers.CDataMain import CDataMain
from Experiments.CConfig import CConfig
from Experiments.ExecuteExperiments.Helpers.CResultsStore import CResultsStore
from Experiments.ExecuteExperiments.Helpers.CDistanceAnalysis import CDistanceAnalysis


def run_algorithms_on_synthetic_data(All_C_hat_u:dict,algID:int):     
    # Step 2: Create Test Data Main Instance
    testData = CDataMain(All_C_hat_u)
    embeddingDimensions = CConfig.EMBEDDING_DIMENSIONS
    if(algID == 2):
        (MAT_E, loss_for_each_user) = Alg_2_AutoEncoder(
                embeddingDimensions=embeddingDimensions,
                testData=testData)
    elif(algID == 3):
        (MAT_E, loss_for_each_user) = Alg_3_MatrixFactorization(
                embeddingDimensions=embeddingDimensions,
                testData=testData)
    else:
        raise ValueError(f"Unsupported algorithm ID: {algID}")
    print(f"Generated Embeddings with Algorithm {algID}") 
    return (MAT_E, loss_for_each_user)

def run_pca_on_synthetic_data(All_C_hat_u:dict,algID:int):     
    testData = CDataMain(All_C_hat_u)
    embeddingDimensions = CConfig.EMBEDDING_DIMENSIONS
   
    (MAT_E, loss_for_each_user) = Alg_Baseline_PCA(
                embeddingDimensions=embeddingDimensions,
                testData=testData)
    print(f"Generated Embeddings with PCA") 
    return (MAT_E, loss_for_each_user)


def store_results_in_file(MAT_E, loss_for_each_user, algID:int):
    store = CResultsStore(algID=algID)
    # Store embeddings in file
    store.store_embeddings(MAT_E)
    print(f"Agent embeddings stored in file")
    # Store training loss in file
    store.store_training_loss(loss_for_each_user)
    print(f"Training loss stored in file")


# Run the experiments and store embeddings and losses in file
if __name__== "__main__":
    print(f"Starting experiments on synthetic data at time {datetime.datetime.now()}...")
    print("Running A/g #1 on synthetic data...")
    # Step 1: Data Preparation
    All_C_hat_u = Algorithm_1_DataPreparation()

    algorithmsIds = [2, 3]
    for algID in algorithmsIds:
        print(f"Computing embeddings for Algorithm {algID}...")
        (MAT_E, loss_for_each_user) = \
            run_algorithms_on_synthetic_data(All_C_hat_u,algID)
        print(f"Storing embeddings for Algorithm {algID}...")
        store_results_in_file(MAT_E, loss_for_each_user, algID=algID)
    print(f"Experiments completed at time {datetime.datetime.now()}.")

       
 