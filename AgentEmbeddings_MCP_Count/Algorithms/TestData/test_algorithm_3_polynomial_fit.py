"""Behavior and input-validation tests for polynomial-fit embeddings."""

import unittest

import numpy as np
import torch

from Algorithms.Alg_3_PolynomialFit import Alg_3_PolynomialFit
from Algorithms.Helpers.CPolynomialFitReduction import CPolynomialFitReduction
from Algorithms.TestData.CTestData_Simple import CTestData_Simple


class TestAlgorithm3PolynomialFit(unittest.TestCase):
    def test_generates_polynomial_embeddings_and_reconstruction_losses(self):
        """Verify valid embeddings, losses, and expected agent similarity."""
        testData = CTestData_Simple()

        embeddings, losses = Alg_3_PolynomialFit(
            embeddingDimensions=2,
            testData=testData,
        )

        self.assertEqual(tuple(embeddings.shape), (4, 2))
        self.assertEqual(tuple(losses.shape), (4,))
        self.assertTrue(torch.isfinite(embeddings).all())
        self.assertTrue(torch.isfinite(losses).all())
        self.assertTrue(torch.all(losses >= 0))

        # Agents 0 and 2 have identical tool usage, while agent 1 is different.
        self.assertTrue(torch.allclose(embeddings[0], embeddings[2]))
        self.assertFalse(torch.allclose(embeddings[0], embeddings[1]))

    def test_rejects_more_dimensions_than_the_number_of_tools(self):
        with self.assertRaisesRegex(ValueError, "number of tools"):
            Alg_3_PolynomialFit(
                embeddingDimensions=5,
                testData=CTestData_Simple(),
            )

    def test_dimension_24_fit_is_full_rank_and_matches_numpy(self):
        reducer = CPolynomialFitReduction(embeddingDimensions=24)
        design_matrix = reducer._torch_design_matrix(64, torch.device("cpu"))
        self.assertEqual(torch.linalg.matrix_rank(design_matrix).item(), 24)

        generator = torch.Generator().manual_seed(42)
        data = torch.rand((3, 64), generator=generator)
        embeddings, losses = reducer.fit_matrix(data)

        for index in range(data.shape[0]):
            expected_embedding, expected_loss = (
                reducer.get_reduced_dimension_polynomial_fit(
                    data[index:index + 1].numpy()
                )
            )
            np.testing.assert_allclose(
                embeddings[index].numpy(),
                expected_embedding,
                rtol=1e-5,
                atol=1e-6,
            )
            self.assertAlmostEqual(
                losses[index].item(),
                expected_loss,
                places=7,
            )


if __name__ == "__main__":
    unittest.main()
