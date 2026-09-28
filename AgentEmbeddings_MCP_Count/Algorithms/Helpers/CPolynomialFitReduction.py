import numpy as np
import torch


class CPolynomialFitReduction:
    def __init__(self,embeddingDimensions:int=8):
        self.embeddingDimensions = embeddingDimensions

    def _numpy_design_matrix(self, number_of_tools: int):
        x = np.linspace(-1.0, 1.0, number_of_tools)
        return np.polynomial.chebyshev.chebvander(
            x,
            self.embeddingDimensions - 1,
        )

    def _torch_design_matrix(
        self,
        number_of_tools: int,
        device: torch.device,
    ):
        x = torch.linspace(
            -1.0,
            1.0,
            steps=number_of_tools,
            dtype=torch.float64,
            device=device,
        )
        columns = [torch.ones_like(x)]
        if self.embeddingDimensions > 1:
            columns.append(x)
        for _ in range(2, self.embeddingDimensions):
            columns.append(2.0 * x * columns[-1] - columns[-2])
        return torch.stack(columns, dim=1)
        
    def get_reduced_dimension_polynomial_fit(self, data: np.ndarray):
        no_of_tools = data.shape[1]
        if self.embeddingDimensions > no_of_tools:
            raise ValueError(
                "embeddingDimensions cannot exceed the number of tools"
            )

        y = data.flatten().astype(np.float64, copy=False)
        design_matrix = self._numpy_design_matrix(no_of_tools)
        coefficients, _, _, _ = np.linalg.lstsq(
            design_matrix,
            y,
            rcond=None,
        )
        reconstructed = design_matrix @ coefficients
        loss = float(np.mean(np.square(reconstructed - y)))
        return coefficients, loss

    def fit_matrix(self, data: torch.Tensor, batch_size: int = 256):
        number_of_agents, number_of_tools = data.shape
        if self.embeddingDimensions > number_of_tools:
            raise ValueError(
                "embeddingDimensions cannot exceed the number of tools"
            )

        design_matrix = self._torch_design_matrix(
            number_of_tools,
            data.device,
        )
        projection = torch.linalg.pinv(design_matrix).transpose(0, 1)

        embeddings = torch.empty(
            (number_of_agents, self.embeddingDimensions),
            dtype=data.dtype,
            device=data.device,
        )
        losses = torch.empty(
            number_of_agents,
            dtype=data.dtype,
            device=data.device,
        )

        for start in range(0, number_of_agents, batch_size):
            end = min(start + batch_size, number_of_agents)
            batch = data[start:end].to(dtype=torch.float64)
            batch_embeddings = batch @ projection
            reconstructed = batch_embeddings @ design_matrix.transpose(0, 1)
            embeddings[start:end] = batch_embeddings.to(dtype=data.dtype)
            losses[start:end] = (
                (reconstructed - batch).pow(2).mean(dim=1).to(dtype=data.dtype)
            )

        return embeddings, losses