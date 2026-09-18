"""Behavior and reproducibility tests for the shared autoencoder embeddings."""

import unittest

import torch

from Algorithms.Alg_2_AutoEncoder import (
    Alg_2_AutoEncoder,
)
from Algorithms.TestData.CTestData_Simple import CTestData_Simple


class TestAlgorithm2Autoencoder(unittest.TestCase):
    def test_generates_shared_embeddings_with_expected_behavior(self):
        """Verify shape, reconstruction quality, and behavioral similarity."""
        testData = CTestData_Simple()

        embeddings, losses = Alg_2_AutoEncoder(
            embeddingDimensions=2,
            testData=testData,
        )

        self.assertEqual(tuple(embeddings.shape), (4, 2))
        self.assertEqual(tuple(losses.shape), (4,))
        self.assertTrue(torch.isfinite(embeddings).all())
        self.assertTrue(torch.isfinite(losses).all())
        self.assertLess(losses.mean().item(), 1e-4)
        self.assertLess(losses.max().item(), 5e-4)

        # Users 0 and 2 have identical tool usage, while user 1 is different.
        self.assertTrue(torch.equal(embeddings[0], embeddings[2]))
        self.assertFalse(torch.equal(embeddings[0], embeddings[1]))

    def test_training_is_reproducible(self):
        """Verify that fixed random seeds produce identical training results."""
        testData = CTestData_Simple()

        firstEmbeddings, firstLosses = Alg_2_AutoEncoder(
            embeddingDimensions=2,
            testData=testData,
        )
        secondEmbeddings, secondLosses = Alg_2_AutoEncoder(
            embeddingDimensions=2,
            testData=testData,
        )

        self.assertTrue(torch.equal(firstEmbeddings, secondEmbeddings))
        self.assertTrue(torch.equal(firstLosses, secondLosses))


if __name__ == "__main__":
    unittest.main()
