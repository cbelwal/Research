"""
Generate user embeddings with truncated singular value decomposition.

The user-tool matrix is factorized into user and tool latent factors. The user
factors are returned as embeddings in one shared coordinate system.
"""
import os
import sys

import torch
from sklearn.decomposition import TruncatedSVD

topRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(topRootPath)

from Algorithms.Helpers.IUserToolMatrix import IUserToolMatrix

RANDOM_SEED = 1
SVD_ITERATIONS = 7


def Alg_3_MatrixFactorization(
    embeddingDimensions: int = 8,
    testData: IUserToolMatrix = None,
):
    if testData is None:
        raise ValueError("testData is required")
    if embeddingDimensions <= 0:
        raise ValueError("embeddingDimensions must be greater than zero")

    userToolMatrix = testData.get_MAT_u_tau().float()
    if userToolMatrix.ndim != 2:
        raise ValueError("The user-tool matrix must be two-dimensional")
    if userToolMatrix.shape != (testData.NumberOfUsers, testData.NumberOfTools):
        raise ValueError(
            "The user-tool matrix shape does not match NumberOfUsers and NumberOfTools"
        )
    if not torch.isfinite(userToolMatrix).all():
        raise ValueError("The user-tool matrix must contain only finite values")

    maximumDimensions = min(testData.NumberOfUsers, testData.NumberOfTools)
    if embeddingDimensions > maximumDimensions:
        raise ValueError(
            f"embeddingDimensions cannot exceed the matrix rank bound "
            f"({maximumDimensions})"
        )

    factorizer = TruncatedSVD(
        n_components=embeddingDimensions,
        n_iter=SVD_ITERATIONS,
        random_state=RANDOM_SEED,
    )
    matrixArray = userToolMatrix.numpy()
    embeddingsArray = factorizer.fit_transform(matrixArray)
    reconstructedArray = factorizer.inverse_transform(embeddingsArray)

    MAT_E = torch.from_numpy(embeddingsArray).float()
    reconstructed = torch.from_numpy(reconstructedArray).float()
    loss_for_each_user = (reconstructed - userToolMatrix).pow(2).mean(dim=1)

    return MAT_E, loss_for_each_user
