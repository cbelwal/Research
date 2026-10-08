"""
Generate agent embeddings by fitting a polynomial to each tool-usage vector.

Chebyshev polynomial coefficients form the embedding. All agents use the same
normalized tool-index domain, so their coefficients are directly comparable
without the numerical instability of a high-degree monomial basis.
"""
import os
import sys

import torch

topRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(topRootPath)

from Algorithms.Helpers.CPolynomialFitReduction import CPolynomialFitReduction
from Algorithms.Helpers.IAgentToolMatrix import IAgentToolMatrix


def Alg_3_PolynomialFit(
    embeddingDimensions: int = 8,
    testData: IAgentToolMatrix = None,
):
    if testData is None:
        raise ValueError("testData is required")
    if embeddingDimensions <= 0:
        raise ValueError("embeddingDimensions must be greater than zero")

    agentToolMatrix = testData.get_MAT_a_tau().float()
    if agentToolMatrix.ndim != 2:
        raise ValueError("The agent-tool matrix must be two-dimensional")
    if agentToolMatrix.shape != (testData.NumberOfAgents, testData.NumberOfTools):
        raise ValueError(
            "The agent-tool matrix shape does not match NumberOfAgents and NumberOfTools"
        )
    if testData.NumberOfAgents <= 0 or testData.NumberOfTools <= 0:
        raise ValueError("The agent-tool matrix must contain agents and tools")
    if embeddingDimensions > testData.NumberOfTools:
        raise ValueError(
            "embeddingDimensions cannot exceed the number of tools"
        )
    if not torch.isfinite(agentToolMatrix).all():
        raise ValueError("The agent-tool matrix must contain only finite values")

    reducer = CPolynomialFitReduction(embeddingDimensions)
    return reducer.fit_matrix(agentToolMatrix)
