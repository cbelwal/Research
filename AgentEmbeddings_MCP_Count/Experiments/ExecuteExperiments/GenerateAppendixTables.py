"""Generate CSV inputs for the agent-embedding comparison tables."""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

topRootPath = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(topRootPath)

from Experiments.CConfig import CConfig


ALGORITHM_NAMES = {2: "Autoencoder", 3: "Polynomial"}
DEFAULT_AGENT_ID = 1
DEFAULT_TOP_K = 5
DEFAULT_MAX_CLUSTERS = 10


@dataclass(frozen=True)
class Experiment:
    dimension: int
    algorithm_id: int
    embeddings: torch.Tensor

    @property
    def algorithm_name(self) -> str:
        return ALGORITHM_NAMES[self.algorithm_id]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_results = (
        script_dir.parent
        / "Data"
        / "ExperimentResults"
        / f"a{CConfig.MAX_AGENTS}"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_results / "AppendixTables",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=script_dir.parent / "Data" / CConfig.DB_FILE_NAME,
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=list(CConfig.EMBEDDING_DIMENSIONS_TO_RUN),
    )
    parser.add_argument("--agent-id", type=int, default=DEFAULT_AGENT_ID)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-clusters", type=int, default=DEFAULT_MAX_CLUSTERS)
    return parser.parse_args()


def load_tensor(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment tensor: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"Expected a two-dimensional tensor in {path}")
    return value.detach().float()


def load_experiments(results_dir: Path, dimensions: list[int]):
    experiments = []
    for dimension in dimensions:
        dimension_dir = results_dir / f"Emb_dim_{dimension}"
        for algorithm_id in ALGORITHM_NAMES:
            path = (
                dimension_dir
                / f"agent_embeddings_a{CConfig.MAX_AGENTS}_alg_{algorithm_id}.pt"
            )
            embeddings = load_tensor(path)
            if embeddings.shape != (CConfig.MAX_AGENTS, dimension):
                raise ValueError(
                    f"{path} has shape {tuple(embeddings.shape)}, expected "
                    f"({CConfig.MAX_AGENTS}, {dimension})"
                )
            experiments.append(Experiment(dimension, algorithm_id, embeddings))
    return experiments


def read_canary_groups(database: Path):
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT agent_id, canary_category FROM canary_agents "
            "ORDER BY canary_category, agent_id"
        ).fetchall()
    groups = {}
    for agent_id, category in rows:
        groups.setdefault(int(category), []).append(int(agent_id))
    return groups


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    minimum = embeddings.amin(dim=0)
    span = (embeddings.amax(dim=0) - minimum).clamp_min(1e-12)
    return F.normalize((embeddings - minimum) / span, p=2, dim=1)


def nearest_agents(embeddings: torch.Tensor, agent_id: int, top_k: int):
    target_index = agent_id - 1
    values = embeddings.numpy()
    target = values[target_index : target_index + 1]
    cosine = cdist(target, values, metric="cosine")[0]
    euclidean = cdist(target, values, metric="euclidean")[0]
    cosine[target_index] = np.inf
    euclidean[target_index] = np.inf
    agent_ids = np.arange(1, embeddings.shape[0] + 1)
    return {
        "Cosine": agent_ids[np.lexsort((agent_ids, cosine))[:top_k]].tolist(),
        "Euclidean": agent_ids[
            np.lexsort((agent_ids, euclidean))[:top_k]
        ].tolist(),
    }


def sampled_pairwise_mean_std(
    embeddings: torch.Tensor,
    metric: str,
    max_pairs: int,
    seed: int,
):
    count = embeddings.shape[0]
    total_pairs = count * (count - 1) // 2
    sample_count = min(max_pairs, total_pairs)
    generator = torch.Generator().manual_seed(seed)
    left_indices = torch.randint(0, count, (sample_count,), generator=generator)
    right_indices = torch.randint(
        0,
        count - 1,
        (sample_count,),
        generator=generator,
    )
    right_indices += (right_indices >= left_indices).long()
    left = embeddings[left_indices]
    right = embeddings[right_indices]
    if metric == "Cosine":
        distances = 1.0 - F.cosine_similarity(left, right, dim=1)
    else:
        distances = torch.linalg.vector_norm(left - right, dim=1)
    return (
        float(distances.mean()),
        float(distances.std(unbiased=False)),
        sample_count,
    )


def elbow_and_silhouette(embeddings: torch.Tensor, max_clusters: int):
    values = embeddings.numpy()
    cluster_counts = list(range(2, min(max_clusters + 1, len(values))))
    inertias = []
    models = {}
    for cluster_count in cluster_counts:
        model = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init="auto",
        ).fit(values)
        models[cluster_count] = model
        inertias.append(float(model.inertia_))
    knee = KneeLocator(
        cluster_counts,
        inertias,
        curve="convex",
        direction="decreasing",
    ).elbow
    optimal_clusters = int(knee) if knee is not None else 2
    labels = models[optimal_clusters].labels_
    sample_size = min(CConfig.SILHOUETTE_SAMPLE_SIZE, len(values))
    score = silhouette_score(
        values,
        labels,
        sample_size=sample_size,
        random_state=42,
    )
    return optimal_clusters, float(score)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main():
    args = parse_args()
    experiments = load_experiments(args.results_dir, args.dimensions)
    canary_groups = read_canary_groups(args.database)

    nearest_rows = []
    for experiment in experiments:
        neighbors = nearest_agents(
            experiment.embeddings,
            args.agent_id,
            args.top_k,
        )
        for metric, agent_ids in neighbors.items():
            row = {
                "Dimension": experiment.dimension,
                "Distance": metric,
                "Algorithm": experiment.algorithm_name,
            }
            row.update(
                {
                    str(rank): agent_id
                    for rank, agent_id in enumerate(agent_ids, start=1)
                }
            )
            nearest_rows.append(row)
    write_csv(
        args.output_dir / "table_a1_nearest_agents.csv",
        ["Dimension", "Distance", "Algorithm"]
        + [str(rank) for rank in range(1, args.top_k + 1)],
        nearest_rows,
    )

    distance_rows = []
    for experiment in experiments:
        prepared = normalize_embeddings(experiment.embeddings)
        groups = [
            ("Canary 1", [value - 1 for value in canary_groups[1]]),
            ("Canary 2", [value - 1 for value in canary_groups[2]]),
            ("All agents", list(range(prepared.shape[0]))),
        ]
        for group_name, indices in groups:
            group = prepared[indices]
            for metric in ("Cosine", "Euclidean"):
                mean, std, pair_count = sampled_pairwise_mean_std(
                    group,
                    metric,
                    CConfig.MAX_PAIR_SAMPLES,
                    seed=42,
                )
                distance_rows.append(
                    {
                        "Dimension": experiment.dimension,
                        "Group": group_name,
                        "Distance": metric,
                        "Algorithm": experiment.algorithm_name,
                        "Mean": mean,
                        "SD": std,
                        "Pairs": pair_count,
                    }
                )
    write_csv(
        args.output_dir / "table_a2_pairwise_distances.csv",
        ["Dimension", "Group", "Distance", "Algorithm", "Mean", "SD", "Pairs"],
        distance_rows,
    )

    clustering_rows = []
    for experiment in experiments:
        optimal_clusters, score = elbow_and_silhouette(
            experiment.embeddings,
            args.max_clusters,
        )
        clustering_rows.append(
            {
                "Dimension": experiment.dimension,
                "Algorithm": experiment.algorithm_name,
                "Optimal Clusters": optimal_clusters,
                "Silhouette Score": score,
                "Silhouette Sample Size": min(
                    CConfig.SILHOUETTE_SAMPLE_SIZE,
                    experiment.embeddings.shape[0],
                ),
            }
        )
    write_csv(
        args.output_dir / "table_a3_clustering.csv",
        [
            "Dimension",
            "Algorithm",
            "Optimal Clusters",
            "Silhouette Score",
            "Silhouette Sample Size",
        ],
        clustering_rows,
    )


if __name__ == "__main__":
    main()
