"""Non-negative matrix factorization baseline with generalized KL divergence."""

import time

import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import MiniBatchNMF, NMF

from Algorithms.Helpers.IAgentToolMatrix import IAgentToolMatrix


def fit_nmf_kl(
    matrix,
    embedding_dimensions: int = 8,
    max_iter: int = 300,
    random_state: int = 42,
):
    """Fit NMF-KL and return float32 embeddings, elapsed seconds, and the model."""
    if embedding_dimensions <= 0:
        raise ValueError("embedding_dimensions must be positive")
    if embedding_dimensions > min(matrix.shape):
        raise ValueError(
            "embedding_dimensions cannot exceed the smaller matrix dimension"
        )

    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    if sparse.issparse(matrix):
        if matrix.data.size and np.min(matrix.data) < 0:
            raise ValueError("NMF-KL requires non-negative input values")
    else:
        matrix = np.asarray(matrix)
        if not np.isfinite(matrix).all():
            raise ValueError("NMF-KL input must contain only finite values")
        if np.min(matrix) < 0:
            raise ValueError("NMF-KL requires non-negative input values")

    started = time.perf_counter()
    if matrix.shape[0] >= 1000:
        model = MiniBatchNMF(
            n_components=embedding_dimensions,
            init="nndsvda",
            beta_loss="kullback-leibler",
            max_iter=max_iter,
            batch_size=1024,
            random_state=random_state,
            transform_max_iter=max_iter,
            tol=1.0e-4,
        )
    else:
        model = NMF(
            n_components=embedding_dimensions,
            init="nndsvda",
            solver="mu",
            beta_loss="kullback-leibler",
            max_iter=max_iter,
            random_state=random_state,
            tol=1.0e-4,
        )

    embeddings = model.fit_transform(matrix)
    elapsed_seconds = time.perf_counter() - started
    return embeddings.astype(np.float32), elapsed_seconds, model


def Alg_Baseline_NMF_KL(
    embeddingDimensions: int = 8,
    testData: IAgentToolMatrix = None,
    maxIter: int = 300,
    randomState: int = 42,
):
    """Generate NMF-KL embeddings using the repository algorithm interface."""
    if testData is None:
        raise ValueError("testData is required")

    matrix = testData.get_MAT_a_tau()
    embeddings, _, _ = fit_nmf_kl(
        matrix,
        embedding_dimensions=embeddingDimensions,
        max_iter=maxIter,
        random_state=randomState,
    )
    losses = torch.zeros(matrix.shape[0], dtype=torch.float32)
    return torch.from_numpy(embeddings), losses
