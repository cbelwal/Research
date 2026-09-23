"""
Embeddings are generated using a Polynomial reduction

Since this uses the PyTorch library, the sequence of steps will vary with the paper

"""
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))
sys.path.append(topRootPath)
#----------------------------------------------
from Algorithms.Helpers.IAgentToolMatrix import IAgentToolMatrix

SCALING_FACTOR = 1.0

# Reduces values to the given dimension usine PCA
def __generate_pca_for_given_values__(values, reduce_to_dim:int=8):
    # Convert embeddings dict to a 2D array for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(values)
    model = PCA(n_components=reduce_to_dim)
    # Fit PCA to the scaled data and transform it
    pca_components = model.fit_transform(X_scaled)
    return pca_components

# Mainly for evaluation baseline via PCA
def Alg_Baseline_PCA(embeddingDimensions:int=8,
                                       testData: IAgentToolMatrix = None):
    MAT_a_tau = testData.get_MAT_a_tau()

    # Multiply all values by a scaling factor to improve training stability
    MAT_a_tau = MAT_a_tau * SCALING_FACTOR

    print("Starting Model Training")
    
    MAT_E = torch.zeros(testData.NumberOfAgents, embeddingDimensions)
    loss_for_each_agent = torch.zeros(testData.NumberOfAgents)
    # Train the model for each agent
    # For PCA we need to validate all samples together so we 
    # cannot use per sample
    #for i in tqdm(range(0,testData.NumberOfAgents)):
        
    # the elements will be the same, just the shape will be different
    tmpMAT_a_tau = MAT_a_tau #.view(1, testData.NumberOfTools)
        
    # Convert tmpMAT_a_tau to numpy
    num_py_array = tmpMAT_a_tau.numpy()
        
    np_array = __generate_pca_for_given_values__(num_py_array,
                                                    reduce_to_dim=embeddingDimensions)
                                                      
        # Convert back to tensor
    MAT_E = torch.tensor(np_array, dtype=torch.float32)
    # loss_for_each_agent will be 0
    #loss_for_each_agent = torch.tensor(0., dtype=torch.float32)

                    
    return (MAT_E,loss_for_each_agent)

   
