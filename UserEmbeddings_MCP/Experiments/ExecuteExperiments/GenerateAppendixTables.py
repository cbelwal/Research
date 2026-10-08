"""Generate CSV inputs for appendix Tables A1, A2, and A3.

The script consumes the saved tensors under:
    Experiments/Data/ExperimentResults/u10000/Emb_dim_{8,32}

It does not retrain any model. When the synthetic-data SQLite database is not
available, the canary groups are reconstructed from the original seeded data
generation procedure.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


ALGORITHM_NAMES = {
    2: "2",
    3: "3",
    11: "PCA",
    21: "Raw",
}
LEARNED_ALGORITHM_IDS = {2, 3}
DEFAULT_DIMENSIONS = (8, 32)
DEFAULT_USER_ID = 1
DEFAULT_TOP_K = 5
DEFAULT_MAX_CLUSTERS = 10
CANARY_PERCENTAGE = 5
SYNTHETIC_DATA_SEED = 42
SYNTHETIC_MCP_SERVERS = 100
MIN_TOOLS_PER_MCP_SERVER = 1
MAX_TOOLS_PER_MCP_SERVER = 50
RAW_ZERO_SENTINEL = 1e-4


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
    default_results = script_dir.parent / "Data" / "ExperimentResults" / "u10000"
    default_output = default_results / "AppendixTables"
    default_database = script_dir.parent / "Data" / "mcp_interactions_u10000.db"

    parser = argparse.ArgumentParser(
        description="Generate CSV data for appendix Tables A1, A2, and A3."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results,
        help="Directory containing Emb_dim_<N> experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory in which the three CSV files will be written.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=default_database,
        help="Optional synthetic-data SQLite database used to read canary users.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=list(DEFAULT_DIMENSIONS),
        help="Embedding dimensions to include.",
    )
    parser.add_argument("--user-id", type=int, default=DEFAULT_USER_ID)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-clusters", type=int, default=DEFAULT_MAX_CLUSTERS)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for pairwise distance calculations.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tensor(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment tensor: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"Expected a two-dimensional tensor in {path}")
    return value.detach().to(dtype=torch.float32, device="cpu")


def load_experiments(results_dir: Path, dimensions: Iterable[int]) -> list[Experiment]:
    experiments = []
    expected_users = None
    for dimension in dimensions:
        dimension_dir = results_dir / f"Emb_dim_{dimension}"
        for algorithm_id in ALGORITHM_NAMES:
            path = dimension_dir / f"user_embeddings_u10000_alg_{algorithm_id}.pt"
            embeddings = load_tensor(path)
            if algorithm_id != 21 and embeddings.shape[1] != dimension:
                raise ValueError(
                    f"{path} has dimension {embeddings.shape[1]}, expected {dimension}"
                )
            if expected_users is None:
                expected_users = embeddings.shape[0]
            elif embeddings.shape[0] != expected_users:
                raise ValueError(f"{path} has an inconsistent user count")
            experiments.append(Experiment(dimension, algorithm_id, embeddings))
    return experiments


def read_canary_groups(database: Path) -> dict[int, list[int]] | None:
    if not database.is_file() or database.stat().st_size == 0:
        return None
    with sqlite3.connect(database) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='canary_users'"
        ).fetchone()
        if table is None:
            return None
        rows = connection.execute(
            "SELECT user_id, canary_category "
            "FROM canary_users ORDER BY canary_category, user_id"
        ).fetchall()
    groups: dict[int, list[int]] = {}
    for user_id, category in rows:
        groups.setdefault(int(category), []).append(int(user_id))
    return groups if 1 in groups and 2 in groups else None


def reconstruct_canary_groups(num_users: int) -> dict[int, list[int]]:
    count = math.ceil(CANARY_PERCENTAGE / 100 * num_users) + 1
    midpoint = math.ceil(num_users / 2)
    rng = random.Random(SYNTHETIC_DATA_SEED)
    # Data generation creates MCP servers before selecting canary users. Advance
    # the RNG through those draws to reproduce the original group membership.
    for _ in range(SYNTHETIC_MCP_SERVERS):
        rng.randint(MIN_TOOLS_PER_MCP_SERVER, MAX_TOOLS_PER_MCP_SERVER)
    canary_1 = sorted(rng.sample(range(1, midpoint), count))
    canary_2 = sorted(rng.sample(range(midpoint + 1, num_users), count))
    return {1: canary_1, 2: canary_2}


def prepare_for_table_a2(experiment: Experiment) -> torch.Tensor:
    embeddings = experiment.embeddings
    if experiment.algorithm_id in LEARNED_ALGORITHM_IDS:
        minimum = embeddings.amin(dim=0)
        span = (embeddings.amax(dim=0) - minimum).clamp_min(1e-12)
        embeddings = (embeddings - minimum) / span
    elif experiment.algorithm_id == 21:
        # CDataMain stores absent sparse entries as 1e-4. Table A2 was computed
        # from the original sparse raw counts, where those entries are zero.
        embeddings = embeddings.masked_fill(
            embeddings <= RAW_ZERO_SENTINEL + torch.finfo(embeddings.dtype).eps,
            0.0,
        )
    return F.normalize(embeddings, p=2, dim=1)


def nearest_users(
    embeddings: torch.Tensor, user_id: int, top_k: int
) -> dict[str, list[int]]:
    if not 1 <= user_id <= embeddings.shape[0]:
        raise ValueError(f"User ID {user_id} is outside the available user range")
    if top_k >= embeddings.shape[0]:
        raise ValueError("top-k must be smaller than the number of users")

    target_index = user_id - 1
    values = embeddings.numpy()
    target = values[target_index : target_index + 1]
    # SciPy accumulates in float64 here. That matters for Algorithm 2, whose
    # cosine distances are extremely close and otherwise collapse into ties.
    cosine = cdist(target, values, metric="cosine")[0]
    euclidean = cdist(target, values, metric="euclidean")[0]
    cosine[target_index] = np.inf
    euclidean[target_index] = np.inf

    user_ids = np.arange(1, embeddings.shape[0] + 1)
    return {
        "Cosine": user_ids[
            np.lexsort((user_ids, cosine))[:top_k]
        ].tolist(),
        "Euclidean": user_ids[
            np.lexsort((user_ids, euclidean))[:top_k]
        ].tolist(),
    }


def pairwise_mean_std(
    embeddings: torch.Tensor,
    metric: str,
    device: torch.device,
    block_size: int = 1024,
) -> tuple[float, float]:
    """Compute population mean/SD over unique pairs without storing all distances."""
    values = embeddings.to(device)
    count = 0
    total = 0.0
    total_squared = 0.0

    for start in range(0, values.shape[0], block_size):
        left = values[start : start + block_size]

        if left.shape[0] > 1:
            within = torch.pdist(left, p=2)
            if metric == "Cosine":
                within = 0.5 * within.square()
            within = within.to(dtype=torch.float64)
            count += within.numel()
            total += within.sum().item()
            total_squared += within.square().sum().item()

        for other_start in range(start + block_size, values.shape[0], block_size):
            right = values[other_start : other_start + block_size]
            dots = left @ right.T
            if metric == "Cosine":
                distances = (1.0 - dots).clamp(min=0.0, max=2.0)
            else:
                distances = (2.0 - 2.0 * dots).clamp_min(0.0).sqrt()
            distances = distances.to(dtype=torch.float64)
            count += distances.numel()
            total += distances.sum().item()
            total_squared += distances.square().sum().item()

    mean = total / count
    variance = max(0.0, total_squared / count - mean * mean)
    return mean, math.sqrt(variance)


def elbow_and_silhouette(
    embeddings: torch.Tensor, max_clusters: int
) -> tuple[int, float]:
    values = embeddings.numpy()
    cluster_counts = list(range(2, min(max_clusters + 1, values.shape[0])))
    inertias = []
    for cluster_count in cluster_counts:
        model = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init="auto",
        ).fit(values)
        inertias.append(float(model.inertia_))

    knee = KneeLocator(
        cluster_counts,
        inertias,
        curve="convex",
        direction="decreasing",
    ).elbow
    optimal_clusters = int(knee) if knee is not None else 2
    labels = KMeans(
        n_clusters=optimal_clusters,
        random_state=42,
        n_init=10,
    ).fit_predict(values)
    return optimal_clusters, float(silhouette_score(values, labels))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def generate_table_a1(
    experiments: list[Experiment], user_id: int, top_k: int
) -> list[dict]:
    rows = []
    for experiment in experiments:
        neighbors = nearest_users(experiment.embeddings, user_id, top_k)
        for metric in ("Cosine", "Euclidean"):
            row = {
                "Size": experiment.dimension,
                "Distance": metric,
                "Algorithm": experiment.algorithm_name,
            }
            row.update(
                {
                    str(rank): neighbor
                    for rank, neighbor in enumerate(neighbors[metric], start=1)
                }
            )
            rows.append(row)
    return rows


def generate_table_a2(
    experiments: list[Experiment],
    canary_groups: dict[int, list[int]],
    device: torch.device,
) -> list[dict]:
    by_dimension = {
        dimension: {
            experiment.algorithm_id: experiment
            for experiment in experiments
            if experiment.dimension == dimension
        }
        for dimension in sorted({item.dimension for item in experiments})
    }
    rows = []
    for dimension, algorithms in by_dimension.items():
        prepared = {
            algorithm_id: prepare_for_table_a2(experiment)
            for algorithm_id, experiment in algorithms.items()
        }
        groups = [
            ("Canary 1", [user_id - 1 for user_id in canary_groups[1]]),
            ("Canary 2", [user_id - 1 for user_id in canary_groups[2]]),
            ("All users", list(range(next(iter(prepared.values())).shape[0]))),
        ]
        for group_name, indices in groups:
            for metric in ("Cosine", "Euclidean"):
                row = {
                    "Size": dimension,
                    "Group": group_name,
                    "Distance": metric,
                }
                index_tensor = torch.tensor(indices, dtype=torch.long)
                for algorithm_id, name in ALGORITHM_NAMES.items():
                    mean, std = pairwise_mean_std(
                        prepared[algorithm_id].index_select(0, index_tensor),
                        metric,
                        device,
                    )
                    row[f"{name} Mean"] = mean
                    row[f"{name} SD"] = std
                rows.append(row)
    return rows


def generate_table_a3(
    experiments: list[Experiment], max_clusters: int
) -> list[dict]:
    rows = []
    for experiment in experiments:
        optimal_clusters, score = elbow_and_silhouette(
            experiment.embeddings, max_clusters
        )
        rows.append(
            {
                "Size": experiment.dimension,
                "Algorithm": experiment.algorithm_name,
                "Optimal Clusters": optimal_clusters,
                "Silhouette Score": score,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    experiments = load_experiments(args.results_dir, args.dimensions)
    num_users = experiments[0].embeddings.shape[0]
    canary_groups = read_canary_groups(args.database)
    if canary_groups is None:
        canary_groups = reconstruct_canary_groups(num_users)
        print("Database unavailable; reconstructed canary groups with seed 42.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    table_a1 = generate_table_a1(experiments, args.user_id, args.top_k)
    write_csv(
        args.output_dir / "table_a1_nearest_users.csv",
        ["Size", "Distance", "Algorithm"]
        + [str(rank) for rank in range(1, args.top_k + 1)],
        table_a1,
    )

    table_a2 = generate_table_a2(experiments, canary_groups, device)
    a2_fields = ["Size", "Group", "Distance"]
    for name in ALGORITHM_NAMES.values():
        a2_fields.extend([f"{name} Mean", f"{name} SD"])
    write_csv(
        args.output_dir / "table_a2_pairwise_distances.csv",
        a2_fields,
        table_a2,
    )

    table_a3 = generate_table_a3(experiments, args.max_clusters)
    write_csv(
        args.output_dir / "table_a3_clustering.csv",
        ["Size", "Algorithm", "Optimal Clusters", "Silhouette Score"],
        table_a3,
    )


if __name__ == "__main__":
    main()
