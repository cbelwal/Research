"""Run the focused experiments requested during AAMAS review.

The script intentionally limits scope to the recurring review concerns:

* NMF with KL divergence on synthetic and production agent--tool matrices;
* PCA and normalized raw-count baselines on the production slice;
* a Canary same-cohort retrieval task;
* polynomial tool-index sensitivity and numerical conditioning; and
* a compact autoencoder seed-sensitivity check.

The synthetic clustering experiment uses all agents. Expensive retrieval,
permutation, and autoencoder-seed analyses use fixed, explicitly reported
subsets so they can be reproduced during the rebuttal period.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from kneed import KneeLocator
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from Algorithms.Alg_Baseline_NMF_KL import fit_nmf_kl


METHOD_ORDER = ["Autoencoder", "Polynomial", "PCA", "Raw", "NMF"]
DIMENSIONS = (8, 24)
CLUSTER_COUNTS = tuple(range(2, 11))
CANARY_SEED = 42
MISSING_VALUE = 1.0e-4


@dataclass
class ClusteringResult:
    dataset: str
    dimension: int
    method: str
    selected_clusters: int
    silhouette: float
    sample_size: int
    fit_seconds: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-db", type=Path, required=True)
    parser.add_argument("--production-csv", type=Path, required=True)
    parser.add_argument("--production-agent-map", type=Path, required=True)
    parser.add_argument("--production-tool-map", type=Path, required=True)
    parser.add_argument("--production-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retrieval-background", type=int, default=5000)
    parser.add_argument("--permutation-sample", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--seed-sample", type=int, default=1000)
    parser.add_argument("--seed-epochs", type=int, default=50)
    parser.add_argument("--retrieval-epochs", type=int, default=100)
    parser.add_argument("--nmf-max-iter", type=int, default=300)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)


def load_synthetic_matrix(database: Path):
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(database)
    number_of_agents = int(
        connection.execute("SELECT MAX(id) FROM agents").fetchone()[0]
    )
    number_of_tools = int(
        connection.execute("SELECT MAX(id) FROM mcp_tools").fetchone()[0]
    )
    connection.execute("PRAGMA temp_store = FILE")
    started = time.monotonic()
    last_report = [started]

    def report_progress():
        now = time.monotonic()
        if now - last_report[0] >= 60:
            print(
                "Synthetic aggregation still running "
                f"({(now - started) / 60:.0f} minutes elapsed)...",
                flush=True,
            )
            last_report[0] = now
        return 0

    connection.set_progress_handler(report_progress, 1_000_000)
    rows, columns, values = [], [], []
    cursor = connection.execute(
        """
        SELECT s.agent_id, si.tool_id, COUNT(*) AS call_count
        FROM session_interactions AS si
        JOIN sessions AS s ON s.id = si.session_id
        WHERE si.tool_id != -1
        GROUP BY s.agent_id, si.tool_id
        ORDER BY s.agent_id, si.tool_id
        """
    )
    for agent_id, tool_id, count in cursor:
        rows.append(int(agent_id) - 1)
        columns.append(int(tool_id) - 1)
        values.append(float(count))
    connection.set_progress_handler(None, 0)
    raw = sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(number_of_agents, number_of_tools),
        dtype=np.float32,
    )
    row_totals = np.asarray(raw.sum(axis=1)).ravel()
    positive_rows = row_totals > 0
    inverse_totals = np.zeros_like(row_totals)
    inverse_totals[positive_rows] = 1.0 / row_totals[positive_rows]
    normalized = sparse.diags(inverse_totals) @ raw
    canary_rows = connection.execute(
        "SELECT agent_id, canary_category FROM canary_agents "
        "ORDER BY canary_category, agent_id"
    ).fetchall()
    connection.close()
    canary_groups: dict[int, list[int]] = {}
    for agent_id, category in canary_rows:
        canary_groups.setdefault(int(category), []).append(int(agent_id) - 1)
    print(
        f"Synthetic matrix: {number_of_agents} agents x "
        f"{number_of_tools} tools, {raw.nnz} nonzero cells, "
        f"{np.count_nonzero(~positive_rows)} zero-call agents",
        flush=True,
    )
    return raw, normalized.tocsr(), canary_groups


def load_production_matrix(
    csv_path: Path,
    agent_map_path: Path,
    tool_map_path: Path,
):
    agents = pd.read_csv(agent_map_path)
    agents = agents[
        agents.agent.astype(str).str.startswith("a365-preview-070:")
    ].copy()
    agents = agents[agents.db_agent_id <= 50].sort_values("db_agent_id")
    tools = pd.read_csv(tool_map_path).sort_values("db_tool_id")
    agent_to_row = {
        value.split(":", 1)[1]: index
        for index, value in enumerate(agents.agent.astype(str))
    }
    tool_to_column = {
        value: index for index, value in enumerate(tools.tool.astype(str))
    }
    matrix = np.zeros(
        (len(agent_to_row), len(tool_to_column)),
        dtype=np.float32,
    )
    usecols = ["agent_id", "tool_name"]
    for chunk in pd.read_csv(
        csv_path,
        compression="gzip",
        usecols=usecols,
        keep_default_na=False,
        chunksize=250_000,
    ):
        chunk = chunk[
            chunk.agent_id.isin(agent_to_row)
            & chunk.tool_name.isin(tool_to_column)
        ]
        grouped = (
            chunk.groupby(["agent_id", "tool_name"], sort=False)
            .size()
            .reset_index(name="count")
        )
        for row in grouped.itertuples(index=False):
            matrix[
                agent_to_row[row.agent_id],
                tool_to_column[row.tool_name],
            ] += row.count
    expected_counts = agents.calls.to_numpy(dtype=np.float32)
    observed_counts = matrix.sum(axis=1)
    if not np.array_equal(observed_counts, expected_counts):
        mismatches = np.flatnonzero(observed_counts != expected_counts)
        raise ValueError(
            "Production matrix does not match the published slice for "
            f"{len(mismatches)} agents"
        )
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    print(
        f"Production matrix: {matrix.shape[0]} agents x "
        f"{matrix.shape[1]} tools, {int(matrix.sum())} calls",
        flush=True,
    )
    return matrix, normalized


def load_tensor(path: Path) -> np.ndarray:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Expected tensor in {path}")
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def load_production_embeddings(results: Path, dimension: int):
    dimension_dir = results / f"Emb_dim_{dimension}"
    autoencoder = load_tensor(
        dimension_dir / "agent_embeddings_a56_alg_2.pt"
    )[:50]
    polynomial = load_tensor(
        dimension_dir / "agent_embeddings_a56_alg_3.pt"
    )[:50]
    return autoencoder, polynomial


def chebyshev_embeddings(matrix: np.ndarray, dimension: int):
    number_of_tools = matrix.shape[1]
    x = np.linspace(-1.0, 1.0, number_of_tools)
    design = np.polynomial.chebyshev.chebvander(x, dimension - 1)
    projection = np.linalg.pinv(design).T
    return (matrix @ projection).astype(np.float32), design


def pca_embeddings(matrix: np.ndarray, dimension: int):
    scaled = StandardScaler().fit_transform(matrix)
    return PCA(
        n_components=dimension,
        svd_solver="randomized",
        random_state=42,
    ).fit_transform(scaled).astype(np.float32)


def select_clusters(
    embeddings: np.ndarray,
    seed: int = 42,
    sample_size: int = 5000,
):
    rng = np.random.default_rng(seed)
    if len(embeddings) > sample_size:
        indices = rng.choice(len(embeddings), sample_size, replace=False)
        values = embeddings[indices]
    else:
        values = embeddings
    counts = [
        count for count in CLUSTER_COUNTS if count < len(values)
    ]
    inertias, models = [], {}
    for count in counts:
        model = KMeans(
            n_clusters=count,
            random_state=seed,
            n_init=10,
        ).fit(values)
        models[count] = model
        inertias.append(float(model.inertia_))
    knee = KneeLocator(
        counts,
        inertias,
        curve="convex",
        direction="decreasing",
    ).elbow
    selected = int(knee) if knee is not None else counts[0]
    score = silhouette_score(values, models[selected].labels_)
    return selected, float(score), len(values)


def silhouette_curve(
    embeddings: np.ndarray,
    seed: int = 42,
):
    rows = []
    for count in CLUSTER_COUNTS:
        if count >= len(embeddings):
            continue
        labels = KMeans(
            n_clusters=count,
            random_state=seed,
            n_init=10,
        ).fit_predict(embeddings)
        rows.append(
            {
                "clusters": count,
                "silhouette": float(
                    silhouette_score(embeddings, labels)
                ),
            }
        )
    return rows


class ReviewerAutoencoder(nn.Module):
    def __init__(self, number_of_tools: int, dimension: int):
        super().__init__()
        hidden = min(256, max(dimension * 2, min(number_of_tools, 64)))
        self.encoder = nn.Sequential(
            nn.Linear(number_of_tools, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dimension),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.ReLU(),
            nn.Linear(hidden, number_of_tools),
            nn.Sigmoid(),
        )

    def forward(self, values):
        embeddings = self.encoder(values)
        return self.decoder(embeddings), embeddings


def weighted_loss(reconstructed, target):
    weights = torch.where(target > MISSING_VALUE, 10.0, 1.0)
    return (weights * (reconstructed - target).pow(2)).mean()


def train_autoencoder(
    matrix: np.ndarray,
    dimension: int,
    seed: int,
    epochs: int,
):
    torch.manual_seed(seed)
    values = torch.tensor(matrix, dtype=torch.float32)
    model = ReviewerAutoencoder(values.shape[1], dimension)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-2)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(values),
        batch_size=min(256, len(values)),
        shuffle=True,
        generator=generator,
    )
    started = time.perf_counter()
    final_loss = math.inf
    model.train()
    epochs_run = 0
    for epoch in range(epochs):
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = weighted_loss(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        final_loss = total / len(values)
        epochs_run = epoch + 1
        if final_loss < 5.0e-5:
            break
    model.eval()
    with torch.no_grad():
        _, embeddings = model(values)
    return (
        embeddings.numpy().astype(np.float32),
        final_loss,
        epochs_run,
        time.perf_counter() - started,
    )


def dense_with_missing_values(matrix: sparse.csr_matrix):
    values = matrix.toarray().astype(np.float32)
    values[values == 0] = MISSING_VALUE
    return values


def choose_retrieval_indices(
    number_of_agents: int,
    canary_groups: dict[int, list[int]],
    background_count: int,
):
    canaries = sorted(
        set(canary_groups.get(1, []) + canary_groups.get(2, []))
    )
    canary_set = set(canaries)
    background = np.array(
        [index for index in range(number_of_agents) if index not in canary_set]
    )
    rng = np.random.default_rng(42)
    count = min(background_count, len(background))
    selected_background = rng.choice(background, count, replace=False)
    return np.array(canaries + sorted(selected_background.tolist()))


def retrieval_rows(
    method: str,
    dimension: int,
    embeddings: np.ndarray,
    subset_indices: np.ndarray,
    canary_groups: dict[int, list[int]],
    k: int = 10,
):
    labels = np.zeros(len(subset_indices), dtype=np.int8)
    original_to_subset = {
        original: subset for subset, original in enumerate(subset_indices)
    }
    for category in (1, 2):
        for original in canary_groups[category]:
            labels[original_to_subset[original]] = category
    query_indices = np.flatnonzero(labels > 0)
    normalized = F.normalize(
        torch.tensor(embeddings, dtype=torch.float32),
        p=2,
        dim=1,
    ).numpy()
    neighbors = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ).fit(normalized)
    _, indices = neighbors.kneighbors(normalized[query_indices])
    rows = []
    for category in (1, 2):
        category_queries = query_indices[labels[query_indices] == category]
        positions = {
            query: position for position, query in enumerate(query_indices)
        }
        precisions, recalls, average_precisions = [], [], []
        relevant_total = int(np.sum(labels == category)) - 1
        for query in category_queries:
            ranked = [index for index in indices[positions[query]] if index != query]
            ranked = ranked[:k]
            relevant = (labels[ranked] == category).astype(np.float64)
            hits = int(relevant.sum())
            precisions.append(hits / k)
            recalls.append(hits / relevant_total)
            precision_at_rank = np.cumsum(relevant) / np.arange(1, k + 1)
            average_precisions.append(
                float(np.sum(precision_at_rank * relevant) / min(k, relevant_total))
            )
        rows.append(
            {
                "Dimension": dimension,
                "Method": method,
                "Cohort": f"Canary {category}",
                "Queries": len(category_queries),
                f"Precision@{k}": float(np.mean(precisions)),
                f"Recall@{k}": float(np.mean(recalls)),
                f"mAP@{k}": float(np.mean(average_precisions)),
                "Candidate Agents": len(subset_indices),
            }
        )
    return rows


def top_k_neighbors(values: np.ndarray, query_indices: np.ndarray, k: int):
    normalized = F.normalize(
        torch.tensor(values, dtype=torch.float32),
        p=2,
        dim=1,
    ).numpy()
    model = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ).fit(normalized)
    _, indices = model.kneighbors(normalized[query_indices])
    return [
        [index for index in row if index != query][:k]
        for row, query in zip(indices, query_indices)
    ]


def polynomial_index_sensitivity(
    normalized: sparse.csr_matrix,
    dimension: int,
    sample_size: int,
    permutation_count: int,
):
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        normalized.shape[0],
        min(sample_size, normalized.shape[0]),
        replace=False,
    )
    values = dense_with_missing_values(normalized[sample_indices])
    canonical, _ = chebyshev_embeddings(values, dimension)
    query_indices = rng.choice(
        len(values),
        min(250, len(values)),
        replace=False,
    )
    canonical_neighbors = top_k_neighbors(canonical, query_indices, 10)
    pair_count = min(100_000, len(values) * (len(values) - 1) // 2)
    left = rng.integers(0, len(values), pair_count)
    right = rng.integers(0, len(values) - 1, pair_count)
    right += right >= left
    canonical_norm = F.normalize(
        torch.tensor(canonical), p=2, dim=1
    ).numpy()
    canonical_distance = 1.0 - np.sum(
        canonical_norm[left] * canonical_norm[right],
        axis=1,
    )
    rows = []
    for permutation_number in range(1, permutation_count + 1):
        permutation = np.random.default_rng(
            1000 + permutation_number
        ).permutation(values.shape[1])
        embeddings, _ = chebyshev_embeddings(
            values[:, permutation],
            dimension,
        )
        neighbors = top_k_neighbors(embeddings, query_indices, 10)
        overlaps = [
            len(set(reference) & set(candidate)) / 10
            for reference, candidate in zip(canonical_neighbors, neighbors)
        ]
        normalized_embeddings = F.normalize(
            torch.tensor(embeddings), p=2, dim=1
        ).numpy()
        distances = 1.0 - np.sum(
            normalized_embeddings[left] * normalized_embeddings[right],
            axis=1,
        )
        correlation = spearmanr(
            canonical_distance,
            distances,
        ).statistic
        selected, score, clustering_sample = select_clusters(
            embeddings,
            seed=42,
            sample_size=len(embeddings),
        )
        rows.append(
            {
                "Dimension": dimension,
                "Permutation": permutation_number,
                "Mean Top-10 Neighbor Overlap": float(np.mean(overlaps)),
                "Pairwise Distance Spearman": float(correlation),
                "Selected Clusters": selected,
                "Silhouette": score,
                "Agents": clustering_sample,
                "Sampled Pairs": pair_count,
            }
        )
    return rows


def conditioning_rows(
    normalized: sparse.csr_matrix,
    dimensions: tuple[int, ...],
):
    number_of_tools = normalized.shape[1]
    rows = []
    for dimension in dimensions:
        raw_index = np.arange(number_of_tools, dtype=np.float64)
        scaled_index = np.linspace(-1.0, 1.0, number_of_tools)
        matrices = {
            "Raw-index monomial": np.vander(
                raw_index,
                N=dimension,
                increasing=True,
            ),
            "Scaled monomial": np.vander(
                scaled_index,
                N=dimension,
                increasing=True,
            ),
            "Scaled Chebyshev": np.polynomial.chebyshev.chebvander(
                scaled_index,
                dimension - 1,
            ),
        }
        for basis, design in matrices.items():
            rows.append(
                {
                    "Dimension": dimension,
                    "Tools": number_of_tools,
                    "Basis": basis,
                    "Condition Number": float(np.linalg.cond(design)),
                    "Numerical Rank": int(np.linalg.matrix_rank(design)),
                    "Expected Rank": dimension,
                }
            )
    return rows


def seed_sample_indices(
    number_of_agents: int,
    canary_groups: dict[int, list[int]],
    sample_size: int,
):
    rng = np.random.default_rng(4242)
    each_canary = min(
        sample_size // 4,
        len(canary_groups[1]),
        len(canary_groups[2]),
    )
    selected = []
    for category in (1, 2):
        selected.extend(
            rng.choice(
                canary_groups[category],
                each_canary,
                replace=False,
            ).tolist()
        )
    selected_set = set(selected)
    background = [
        index for index in range(number_of_agents)
        if index not in selected_set
    ]
    selected.extend(
        rng.choice(
            background,
            min(sample_size - len(selected), len(background)),
            replace=False,
        ).tolist()
    )
    return np.array(sorted(selected))


def make_summary_plot(
    path: Path,
    synthetic_clustering: list[dict],
    production_clustering: list[dict],
    retrieval: list[dict],
    sensitivity: list[dict],
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {
        "Autoencoder": "#0078d4",
        "Polynomial": "#8e44ad",
        "PCA": "#d83b01",
        "Raw": "#6b6b6b",
        "NMF": "#107c10",
    }

    syn = pd.DataFrame(synthetic_clustering)
    axes[0, 0].bar(
        [f"NMF-{value}" for value in syn.dimension],
        syn.silhouette,
        color=colors["NMF"],
    )
    axes[0, 0].set_ylabel("Silhouette")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("(a) Synthetic NMF baseline")

    prod = pd.DataFrame(production_clustering)
    prod8 = prod[prod.dimension == 8]
    axes[0, 1].bar(
        prod8.method,
        prod8.silhouette,
        color=[colors[value] for value in prod8.method],
    )
    axes[0, 1].tick_params(axis="x", rotation=30)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel("Silhouette")
    axes[0, 1].set_title("(b) Production baselines, dimension 8")

    ret = pd.DataFrame(retrieval)
    grouped = (
        ret.groupby(["Dimension", "Method"], as_index=False)["mAP@10"]
        .mean()
    )
    for dimension, marker in ((8, "o"), (24, "s")):
        subset = grouped[grouped.Dimension == dimension]
        axes[1, 0].plot(
            subset.Method,
            subset["mAP@10"],
            marker=marker,
            label=f"Dimension {dimension}",
        )
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel("Mean AP@10")
    axes[1, 0].set_title("(c) Canary cohort retrieval")
    axes[1, 0].legend()

    sens = pd.DataFrame(sensitivity)
    for dimension, marker in ((8, "o"), (24, "s")):
        subset = sens[sens.Dimension == dimension]
        axes[1, 1].plot(
            subset.Permutation,
            subset["Mean Top-10 Neighbor Overlap"],
            marker=marker,
            label=f"Dimension {dimension}",
        )
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_xlabel("Random tool-index permutation")
    axes[1, 1].set_ylabel("Top-10 overlap")
    axes[1, 1].set_title("(d) Polynomial index sensitivity")
    axes[1, 1].legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}", flush=True)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_cache = args.output_dir / "synthetic_raw_counts.npz"
    normalized_cache = args.output_dir / "synthetic_normalized_counts.npz"
    canary_cache = args.output_dir / "synthetic_canary_groups.json"
    if (
        raw_cache.is_file()
        and normalized_cache.is_file()
        and canary_cache.is_file()
    ):
        synthetic_raw = sparse.load_npz(raw_cache)
        synthetic_normalized = sparse.load_npz(normalized_cache)
        canary_groups = {
            int(key): value
            for key, value in json.loads(
                canary_cache.read_text(encoding="utf-8")
            ).items()
        }
        print("Loaded cached synthetic matrices.", flush=True)
    else:
        synthetic_raw, synthetic_normalized, canary_groups = (
            load_synthetic_matrix(args.synthetic_db)
        )
        sparse.save_npz(raw_cache, synthetic_raw)
        sparse.save_npz(normalized_cache, synthetic_normalized)
        canary_cache.write_text(
            json.dumps(canary_groups, indent=2),
            encoding="utf-8",
        )
        print("Cached synthetic matrices.", flush=True)
    production_raw, production_normalized = load_production_matrix(
        args.production_csv,
        args.production_agent_map,
        args.production_tool_map,
    )

    synthetic_nmf: dict[int, np.ndarray] = {}
    synthetic_clustering: list[dict] = []
    production_clustering: list[dict] = []
    production_silhouette_curves: list[dict] = []
    production_nmf: dict[int, np.ndarray] = {}
    nmf_timing_cache = args.output_dir / "synthetic_nmf_fit_seconds.json"
    synthetic_nmf_fit_seconds = (
        json.loads(nmf_timing_cache.read_text(encoding="utf-8"))
        if nmf_timing_cache.is_file()
        else {}
    )

    for dimension in DIMENSIONS:
        nmf_cache = args.output_dir / f"synthetic_nmf_dim_{dimension}.npy"
        if nmf_cache.is_file():
            embeddings = np.load(nmf_cache)
            seconds = synthetic_nmf_fit_seconds.get(str(dimension))
            print(f"Loaded {nmf_cache}", flush=True)
        else:
            embeddings, seconds, _ = fit_nmf_kl(
                synthetic_normalized,
                embedding_dimensions=dimension,
                max_iter=args.nmf_max_iter,
            )
            np.save(nmf_cache, embeddings)
            synthetic_nmf_fit_seconds[str(dimension)] = seconds
            nmf_timing_cache.write_text(
                json.dumps(synthetic_nmf_fit_seconds, indent=2),
                encoding="utf-8",
            )
        synthetic_nmf[dimension] = embeddings
        selected, score, sample_size = select_clusters(embeddings)
        synthetic_clustering.append(
            asdict(
                ClusteringResult(
                    "Synthetic",
                    dimension,
                    "NMF",
                    selected,
                    score,
                    sample_size,
                    seconds,
                )
            )
        )

        production_autoencoder, production_polynomial = (
            load_production_embeddings(args.production_results, dimension)
        )
        production_pca = pca_embeddings(
            production_normalized,
            dimension,
        )
        production_nmf[dimension], nmf_seconds, _ = fit_nmf_kl(
            production_normalized,
            embedding_dimensions=dimension,
            max_iter=args.nmf_max_iter,
        )
        representations = {
            "Autoencoder": production_autoencoder,
            "Polynomial": production_polynomial,
            "PCA": production_pca,
            "Raw": production_normalized,
            "NMF": production_nmf[dimension],
        }
        for method in METHOD_ORDER:
            selected, score, sample_size = select_clusters(
                representations[method],
                sample_size=50,
            )
            production_clustering.append(
                asdict(
                    ClusteringResult(
                        "Agent 365",
                        dimension,
                        method,
                        selected,
                        score,
                        sample_size,
                        nmf_seconds if method == "NMF" else None,
                    )
                )
            )
            for curve_point in silhouette_curve(representations[method]):
                production_silhouette_curves.append(
                    {
                        "dataset": "Agent 365",
                        "dimension": dimension,
                        "method": method,
                        **curve_point,
                    }
                )

    write_csv(
        args.output_dir / "synthetic_nmf_clustering.csv",
        synthetic_clustering,
    )
    write_csv(
        args.output_dir / "production_clustering_all_methods.csv",
        production_clustering,
    )
    write_csv(
        args.output_dir / "production_silhouette_curves_all_methods.csv",
        production_silhouette_curves,
    )

    retrieval_indices = choose_retrieval_indices(
        synthetic_normalized.shape[0],
        canary_groups,
        args.retrieval_background,
    )
    retrieval_sparse = synthetic_normalized[retrieval_indices]
    retrieval_matrix = dense_with_missing_values(
        retrieval_sparse
    )
    retrieval_rows_all: list[dict] = []
    autoencoder_retrieval: dict[int, np.ndarray] = {}
    for dimension in DIMENSIONS:
        retrieval_cache = (
            args.output_dir
            / f"retrieval_autoencoder_dim_{dimension}.npy"
        )
        if retrieval_cache.is_file():
            autoencoder = np.load(retrieval_cache)
            final_loss = math.nan
            epochs_run = 0
            seconds = 0.0
            print(f"Loaded {retrieval_cache}", flush=True)
        else:
            autoencoder, final_loss, epochs_run, seconds = train_autoencoder(
                retrieval_matrix,
                dimension,
                seed=1,
                epochs=args.retrieval_epochs,
            )
            np.save(retrieval_cache, autoencoder)
        autoencoder_retrieval[dimension] = autoencoder
        polynomial, _ = chebyshev_embeddings(
            retrieval_matrix,
            dimension,
        )
        pca = pca_embeddings(retrieval_matrix, dimension)
        retrieval_nmf_cache = (
            args.output_dir / f"retrieval_nmf_dim_{dimension}.npy"
        )
        if retrieval_nmf_cache.is_file():
            retrieval_nmf = np.load(retrieval_nmf_cache)
            print(f"Loaded {retrieval_nmf_cache}", flush=True)
        else:
            retrieval_nmf, _, _ = fit_nmf_kl(
                retrieval_sparse,
                embedding_dimensions=dimension,
                max_iter=args.nmf_max_iter,
            )
            np.save(retrieval_nmf_cache, retrieval_nmf)
        representations = {
            "Autoencoder": autoencoder,
            "Polynomial": polynomial,
            "PCA": pca,
            "Raw": retrieval_matrix,
            "NMF": retrieval_nmf,
        }
        for method in METHOD_ORDER:
            retrieval_rows_all.extend(
                retrieval_rows(
                    method,
                    dimension,
                    representations[method],
                    retrieval_indices,
                    canary_groups,
                )
            )
        print(
            f"Retrieval autoencoder dim {dimension}: "
            f"loss={final_loss:.6g}, epochs={epochs_run}, "
            f"seconds={seconds:.1f}",
            flush=True,
        )
    write_csv(
        args.output_dir / "canary_retrieval.csv",
        retrieval_rows_all,
    )

    sensitivity_rows: list[dict] = []
    for dimension in DIMENSIONS:
        sensitivity_rows.extend(
            polynomial_index_sensitivity(
                synthetic_normalized,
                dimension,
                args.permutation_sample,
                args.permutations,
            )
        )
    write_csv(
        args.output_dir / "polynomial_index_sensitivity.csv",
        sensitivity_rows,
    )

    conditioning = conditioning_rows(
        synthetic_normalized,
        DIMENSIONS,
    )
    write_csv(
        args.output_dir / "polynomial_conditioning.csv",
        conditioning,
    )

    stability_indices = seed_sample_indices(
        synthetic_normalized.shape[0],
        canary_groups,
        args.seed_sample,
    )
    stability_matrix = dense_with_missing_values(
        synthetic_normalized[stability_indices]
    )
    stability_rows: list[dict] = []
    for dimension in DIMENSIONS:
        for seed in (1, 2, 3, 4, 5):
            embeddings, loss, epochs_run, seconds = train_autoencoder(
                stability_matrix,
                dimension,
                seed,
                args.seed_epochs,
            )
            selected, score, sample_size = select_clusters(
                embeddings,
                seed=seed,
                sample_size=len(embeddings),
            )
            stability_rows.append(
                {
                    "Dimension": dimension,
                    "Seed": seed,
                    "Agents": sample_size,
                    "Epochs": epochs_run,
                    "Final Weighted MSE": loss,
                    "Selected Clusters": selected,
                    "Silhouette": score,
                    "Fit Seconds": seconds,
                }
            )
    write_csv(
        args.output_dir / "autoencoder_seed_stability.csv",
        stability_rows,
    )

    summary = {
        "synthetic": {
            "agents": synthetic_normalized.shape[0],
            "tools": synthetic_normalized.shape[1],
            "nonzero_cells": synthetic_normalized.nnz,
            "zero_call_agents": int(
                np.count_nonzero(synthetic_normalized.getnnz(axis=1) == 0)
            ),
        },
        "production": {
            "agents": production_normalized.shape[0],
            "tools": production_normalized.shape[1],
            "tool_calls": int(production_raw.sum()),
        },
        "retrieval": {
            "agents": len(retrieval_indices),
            "canary_queries": sum(
                len(canary_groups[value]) for value in (1, 2)
            ),
            "background_agents": args.retrieval_background,
            "epochs": args.retrieval_epochs,
        },
        "permutation": {
            "agents": min(
                args.permutation_sample,
                synthetic_normalized.shape[0],
            ),
            "permutations": args.permutations,
        },
        "seed_stability": {
            "agents": len(stability_indices),
            "seeds": [1, 2, 3, 4, 5],
            "epochs": args.seed_epochs,
        },
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    make_summary_plot(
        args.output_dir / "additional_experiments.svg",
        synthetic_clustering,
        production_clustering,
        retrieval_rows_all,
        sensitivity_rows,
    )


if __name__ == "__main__":
    main()
