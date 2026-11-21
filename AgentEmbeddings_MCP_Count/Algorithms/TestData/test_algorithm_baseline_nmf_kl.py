"""Behavior and validation tests for the NMF-KL baseline."""

import unittest

import numpy as np
import torch

from Algorithms.Alg_Baseline_NMF_KL import (
    Alg_Baseline_NMF_KL,
    fit_nmf_kl,
)
from Algorithms.TestData.CTestData_Simple import CTestData_Simple


class TestAlgorithmBaselineNmfKl(unittest.TestCase):
    def test_generates_deterministic_non_negative_embeddings(self):
        test_data = CTestData_Simple()

        first, losses = Alg_Baseline_NMF_KL(
            embeddingDimensions=2,
            testData=test_data,
            maxIter=100,
        )
        second, _ = Alg_Baseline_NMF_KL(
            embeddingDimensions=2,
            testData=test_data,
            maxIter=100,
        )

        self.assertEqual(tuple(first.shape), (4, 2))
        self.assertEqual(tuple(losses.shape), (4,))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.all(first >= 0))
        self.assertTrue(torch.allclose(first, second))

    def test_rejects_negative_input(self):
        matrix = np.array([[1.0, -1.0], [0.0, 2.0]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            fit_nmf_kl(matrix, embedding_dimensions=1)

    def test_rejects_excessive_dimensions(self):
        matrix = np.ones((2, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "smaller matrix dimension"):
            fit_nmf_kl(matrix, embedding_dimensions=3)


if __name__ == "__main__":
    unittest.main()
