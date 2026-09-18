"""Behavior and input-validation tests for matrix-factorization embeddings."""

import unittest

import torch

from Algorithms.Alg_3_MatrixFactorization import Alg_3_MatrixFactorization
from Algorithms.TestData.CTestData_Simple import CTestData_Simple


class TestAlgorithm3MatrixFactorization(unittest.TestCase):
    def test_generates_shared_embeddings_and_reconstruction_losses(self):
        """Verify valid embeddings, losses, and expected user similarity."""
        testData = CTestData_Simple()

        embeddings, losses = Alg_3_MatrixFactorization(
            embeddingDimensions=2,
            testData=testData,
        )

        self.assertEqual(tuple(embeddings.shape), (4, 2))
        self.assertEqual(tuple(losses.shape), (4,))
        self.assertTrue(torch.isfinite(embeddings).all())
        self.assertTrue(torch.isfinite(losses).all())
        self.assertTrue(torch.all(losses >= 0))

        # Users 0 and 2 have identical tool usage, while user 1 is different.
        self.assertTrue(torch.allclose(embeddings[0], embeddings[2]))
        self.assertFalse(torch.allclose(embeddings[0], embeddings[1]))

    def test_rejects_more_dimensions_than_the_matrix_supports(self):
        """Reject an embedding size larger than the matrix rank bound."""
        with self.assertRaisesRegex(ValueError, "matrix rank bound"):
            Alg_3_MatrixFactorization(
                embeddingDimensions=5,
                testData=CTestData_Simple(),
            )


if __name__ == "__main__":
    unittest.main()
