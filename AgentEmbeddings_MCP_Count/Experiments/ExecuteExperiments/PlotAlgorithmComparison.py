"""Plot sampled silhouette scores for Algorithms 2 and 3."""

import argparse
import os
import sys

import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

topRootPath = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(topRootPath)

from Experiments.CConfig import CConfig
from Experiments.ExecuteExperiments.Helpers.CResultsStore import CResultsStore
from Experiments.Plots.CPlotCommon import CPlotCommon


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=list(CConfig.EMBEDDING_DIMENSIONS_TO_RUN),
    )
    return parser.parse_args()


def load_embeddings(algorithm_id, dimension):
    return CResultsStore(
        algID=algorithm_id,
        embeddingDimensions=dimension,
    ).load_embeddings()


def main():
    args = parse_args()
    cluster_range = list(range(2, 11))
    for dimension in args.dimensions:
        output_directory = os.path.join(
            topRootPath,
            "Experiments",
            "Data",
            "ExperimentResults",
            f"a{CConfig.MAX_AGENTS}",
            f"Emb_dim_{dimension}",
        )
        CPlotCommon.set_output_directory(output_directory)
        series = {}
        for algorithm_id, name in (
            (2, "Algorithm 2 - Autoencoder"),
            (3, "Algorithm 3 - Polynomial"),
        ):
            embeddings = load_embeddings(algorithm_id, dimension).numpy()
            scores = []
            for cluster_count in cluster_range:
                labels = KMeans(
                    n_clusters=cluster_count,
                    random_state=42,
                    n_init="auto",
                ).fit_predict(embeddings)
                scores.append(
                    silhouette_score(
                        embeddings,
                        labels,
                        sample_size=min(
                            CConfig.SILHOUETTE_SAMPLE_SIZE,
                            len(embeddings),
                        ),
                        random_state=42,
                    )
                )
            series[name] = scores
        CPlotCommon.plot_multi_line_xy(
            cluster_range,
            series,
            title="Silhouette Scores vs Number of Clusters - All Algorithms",
            xlabel="Number of Clusters",
            ylabel="Silhouette Score",
            saveFile=True,
        )


if __name__ == "__main__":
    main()
