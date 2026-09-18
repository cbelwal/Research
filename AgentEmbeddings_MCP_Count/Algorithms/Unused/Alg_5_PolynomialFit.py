"""
Unused polynomial-reduction embedding implementation.

Since this uses the PyTorch library, the sequence of steps will vary with the paper

"""
import torch
from tqdm import tqdm

import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))
sys.path.append(topRootPath)
#----------------------------------------------
from Algorithms.Helpers.CPolynomialFitReduction import CPolynomialFitReduction
from Algorithms.Helpers.IUserToolMatrix import IUserToolMatrix  

SCALING_FACTOR = 1.0

def Alg_5_PolynomialFit(embeddingDimensions:int=8,
                                       testData: IUserToolMatrix = None):
    MAT_u_tau = testData.get_MAT_u_tau()

    # Multiply all values by a scaling factor to improve training stability
    MAT_u_tau = MAT_u_tau * SCALING_FACTOR

    print("Starting Model Training")
    
    MAT_E = torch.zeros(testData.NumberOfUsers, embeddingDimensions)
    loss_for_each_user = torch.zeros(testData.NumberOfUsers)
    # Train the model for each user
    for i in tqdm(range(0,testData.NumberOfUsers)):          
        # Model will take the embedding dimenssions as input and return a single output containing value from  tool call.
        model = CPolynomialFitReduction(embeddingDimensions)
        
        # Get the ith row of MAT_tau_u
        # .view: reshape the tensor to be of shape (1, totalNumberOfTools)
        # the elements will be the same, just the shape will be different
        tmpMAT_u_tau = MAT_u_tau[i].view(1, testData.NumberOfTools)
        
        # Convert tmpMAT_u_tau to numpy for processing in CPolynomialFitReduction
        num_py_array = tmpMAT_u_tau.numpy()
        
        (np_array, loss) = model.get_reduced_dimension_polynomial_fit(num_py_array)
        # Convert back to tensor
        MAT_E[i] = torch.tensor(np_array, dtype=torch.float32)
        loss_for_each_user[i] = torch.tensor(loss, dtype=torch.float32)
        #print("Coeffs:",MAT_E[i])
        #break # TEMP: Only 1 user for now
                    
    return (MAT_E,loss_for_each_user)

   
