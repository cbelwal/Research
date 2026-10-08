"""Generate PCA and raw tool-count baselines for each experiment dimension."""

import argparse
import os
import sys

import torch

topRootPath = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(topRootPath)

from Algorithms.Alg_1_DataPreparation import BuildRawAgentToolMatrix
from Algorithms.Alg_Baseline_PCA import Alg_Baseline_PCA
from Algorithms.Helpers.CDataMain import CDataMain
from Experiments.CConfig import CConfig
from Experiments.ExecuteExperiments.Helpers.CResultsStore import CResultsStore
from Experiments.ExecuteExperiments.RunExperiments_Algorithms import (
    build_data_fingerprint,
    build_experiment_metadata,
)

PCA_ALGORITHM_ID = 11
RAW_ALGORITHM_ID = 21


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=list(CConfig.EMBEDDING_DIMENSIONS_TO_RUN),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_normalized_matrix(results_root):
    path = os.path.join(results_root, "agent_tool_matrix.pt")
    try:
        cached = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        cached = torch.load(path, map_location="cpu")
    return cached["matrix"] if isinstance(cached, dict) else cached


def store_baseline(
    algorithm_id,
    dimension,
    embeddings,
    losses,
    metadata,
):
    store = CResultsStore(
        algID=algorithm_id,
        embeddingDimensions=dimension,
    )
    store.store_embeddings(embeddings)
    store.store_training_loss(losses)
    store.store_metadata(metadata)


def results_are_current(store, expected_metadata, force):
    return (
        not force
        and os.path.isfile(
            store.get_file_path(CConfig.BASE_EMBEDDINGS_FILE_NAME)
        )
        and os.path.isfile(
            store.get_file_path(CConfig.BASE_TRAINING_LOSS_FILE_NAME)
        )
        and store.load_metadata() == expected_metadata
    )


def main():
    args = parse_args()
    results_root = os.path.join(
        topRootPath,
        "Experiments",
        "Data",
        "ExperimentResults",
        f"a{CConfig.MAX_AGENTS}",
    )
    normalized_matrix = load_normalized_matrix(results_root)
    data_fingerprint = build_data_fingerprint()
    raw_matrix = BuildRawAgentToolMatrix(
        normalized_matrix,
        cache_path=os.path.join(results_root, "raw_agent_tool_matrix.pt"),
        cache_fingerprint=data_fingerprint,
        force=args.force,
    )

    for dimension in args.dimensions:
        raw_store = CResultsStore(
            algID=RAW_ALGORITHM_ID,
            embeddingDimensions=dimension,
        )
        raw_metadata = build_experiment_metadata(
            data_fingerprint,
            RAW_ALGORITHM_ID,
            dimension,
        )
        if not results_are_current(raw_store, raw_metadata, args.force):
            store_baseline(
                RAW_ALGORITHM_ID,
                dimension,
                raw_matrix,
                torch.zeros(raw_matrix.shape[0]),
                raw_metadata,
            )
            print(f"Stored raw baseline for dimension group {dimension}.")

        pca_store = CResultsStore(
            algID=PCA_ALGORITHM_ID,
            embeddingDimensions=dimension,
        )
        pca_metadata = build_experiment_metadata(
            data_fingerprint,
            PCA_ALGORITHM_ID,
            dimension,
        )
        if not results_are_current(pca_store, pca_metadata, args.force):
            embeddings, losses = Alg_Baseline_PCA(
                embeddingDimensions=dimension,
                testData=CDataMain(raw_matrix),
            )
            store_baseline(
                PCA_ALGORITHM_ID,
                dimension,
                embeddings,
                losses,
                pca_metadata,
            )
            print(f"Stored PCA baseline for dimension {dimension}.")


if __name__ == "__main__":
    main()
