# Computes Cosine and Euclidean distances for L2-normalized embeddings from:
# Algorithms 2, 3, PCA Baseline (11), and Raw Tool Counts (21)
# Each embedding is L2-normalized to unit length before computing distances.
# Distance values are reported as-is (no min-max scaling).
import os, sys
import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.ExecuteExperiments.Helpers.CDistanceFunctions import CDistanceFunctions
from Experiments.ExecuteExperiments.Helpers.CResultsStore import CResultsStore
from Experiments.Database.CDatabaseManager import CDatabaseManager
from Algorithms.Alg_Data_Raw import Algorithm_Data_Raw

# Use L2 normalization on embeddings, report raw distances
class CDistanceAnalysis_Baselines_L2:
    # Algorithms with tensor embeddings
    TENSOR_ALGORITHM_IDS = [2, 3, 11]
    # Raw tool counts algorithm (uses sparse dictionary format)
    RAW_ALG_ID = 21
    # All algorithms for display
    ALL_ALGORITHM_IDS = [2, 3, 11, 21]

    def __init__(self):
        self.dbManager = CDatabaseManager()
        self.canary_users = self.dbManager.get_canary_users()
        self.all_user_ids = self.dbManager.get_all_user_ids()

        # Load tensor embeddings for algorithms 2, 3, and 11
        self.embeddings_by_alg = {}
        for alg_id in self.TENSOR_ALGORITHM_IDS:
            store = CResultsStore(algID=alg_id)
            self.embeddings_by_alg[alg_id] = store.load_embeddings()

        # Load raw tool counts shape for display (lazy load actual data)
        store = CResultsStore(self.RAW_ALG_ID)
        self._raw_tensor_shape = tuple(store.load_embeddings().shape)
        self._raw_tool_counts = None

    def get_canary_user_ids(self, canary_id):
        """Get all user IDs for a given canary category."""
        return self.canary_users.get(canary_id, [])

    def get_raw_tool_counts(self):
        """Lazy load raw tool counts as sparse dictionary from database."""
        if self._raw_tool_counts is None:
            self._raw_tool_counts = Algorithm_Data_Raw()
        return self._raw_tool_counts

    @staticmethod
    def format_value(value: float) -> str:
        """Format value to handle very small numbers using scientific notation."""
        if value == 0:
            return "0.000000"
        elif abs(value) < 0.000001:
            return f"{value:.6e}"
        else:
            return f"{value:.6f}"

    # ==================== L2 NORMALIZATION ====================
    @staticmethod
    def l2_normalize_tensor(MAT_E: torch.Tensor) -> torch.Tensor:
        """
        L2-normalize each row (embedding) in MAT_E so that each has unit length.
        Formula: embedding_norm = embedding / ||embedding||_2

        Args:
            MAT_E: Tensor of shape (num_users, embedding_dim).

        Returns:
            New tensor with each row L2-normalized. Zero-magnitude rows are left as zeros.
        """
        norms = torch.norm(MAT_E, p=2, dim=1, keepdim=True)
        # Avoid division by zero for zero-magnitude rows
        norms = torch.clamp(norms, min=1e-12)
        return MAT_E / norms

    @staticmethod
    def l2_normalize_sparse(vec_dict: dict) -> dict:
        """
        L2-normalize a sparse vector (dictionary).
        Formula: value_norm = value / ||vec||_2

        Args:
            vec_dict: Dictionary mapping key to value (e.g., {tool_id: count}).

        Returns:
            New dictionary with L2-normalized values. Returns empty dict if magnitude is zero.
        """
        magnitude = math.sqrt(sum(v * v for v in vec_dict.values()))
        if magnitude == 0:
            return {}
        return {k: v / magnitude for k, v in vec_dict.items()}

    # ==================== [0,1] MIN-MAX SCALING ====================
    @staticmethod
    def scale_to_unit_range(MAT_E: torch.Tensor) -> torch.Tensor:
        """
        Min-max scale each dimension (column) of MAT_E to the [0, 1] range.
        Formula: scaled = (x - min) / (max - min)

        Args:
            MAT_E: Tensor of shape (num_users, embedding_dim).

        Returns:
            New tensor with each dimension scaled to [0, 1].
            Constant dimensions (max == min) are set to 0.
        """
        col_min = MAT_E.min(dim=0).values
        col_max = MAT_E.max(dim=0).values
        range_vals = col_max - col_min
        # Avoid division by zero for constant dimensions
        range_vals = torch.clamp(range_vals, min=1e-12)
        return (MAT_E - col_min) / range_vals

    # ==================== RAW TOOL COUNT DISTANCE FUNCTIONS ====================
    @staticmethod
    def cosine_distance_sparse(vec1_dict, vec2_dict):
        """
        Compute cosine distance between two sparse vectors (dictionaries).
        Returns 1 - cosine_similarity. Uses high-precision accumulation.
        """
        all_keys = set(vec1_dict.keys()) | set(vec2_dict.keys())

        dot_product = 0.0
        magnitude_1 = 0.0
        magnitude_2 = 0.0

        for key in all_keys:
            val1 = float(vec1_dict.get(key, 0))
            val2 = float(vec2_dict.get(key, 0))
            dot_product += val1 * val2
            magnitude_1 += val1 * val1
            magnitude_2 += val2 * val2

        magnitude_1 = math.sqrt(magnitude_1)
        magnitude_2 = math.sqrt(magnitude_2)

        if magnitude_1 == 0 or magnitude_2 == 0:
            return 1.0  # Maximum distance if one vector is zero

        cosine_similarity = dot_product / (magnitude_1 * magnitude_2)
        # Clamp to [-1, 1] for numerical stability
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
        return 1.0 - cosine_similarity

    @staticmethod
    def euclidean_distance_sparse(vec1_dict, vec2_dict):
        """
        Compute Euclidean distance between two sparse vectors (dictionaries).
        """
        all_keys = set(vec1_dict.keys()) | set(vec2_dict.keys())

        sum_squared_diff = 0.0
        for key in all_keys:
            val1 = vec1_dict.get(key, 0)
            val2 = vec2_dict.get(key, 0)
            sum_squared_diff += (val1 - val2) ** 2

        return math.sqrt(sum_squared_diff)

    # ==================== EMBEDDING DISTANCE FUNCTIONS ====================
    def compute_pairwise_distances_for_user_group(self, user_ids, MAT_E):
        """
        Compute all pairwise cosine and euclidean distances within a group of users
        using tensor embeddings.
        Returns lists of cosine_distances and euclidean_distances.
        """
        cosine_distances = []
        euclidean_distances = []

        for i in range(len(user_ids)):
            user_id_1 = user_ids[i]
            # CAUTION: MAT_E is 0-indexed, user IDs are 1-indexed in DB
            embedding_1 = MAT_E[user_id_1 - 1]

            for j in range(i + 1, len(user_ids)):
                user_id_2 = user_ids[j]
                embedding_2 = MAT_E[user_id_2 - 1]

                cosine_dist = CDistanceFunctions.cosine_distance_tensors(embedding_1, embedding_2)
                euclidean_dist = CDistanceFunctions.euclidean_distance_tensors(embedding_1, embedding_2)

                cosine_distances.append(cosine_dist.item())
                euclidean_distances.append(euclidean_dist.item())

        return cosine_distances, euclidean_distances

    def compute_pairwise_distances_for_user_group_raw(self, user_ids, l2_normalize: bool = True):
        """
        Compute all pairwise cosine and euclidean distances within a group of users
        using raw tool counts (sparse dictionary format).

        Args:
            user_ids: List of user IDs.
            l2_normalize: If True, L2-normalize each sparse vector before computing distances.

        Returns lists of cosine_distances and euclidean_distances.
        """
        raw_counts = self.get_raw_tool_counts()
        cosine_distances = []
        euclidean_distances = []

        # Pre-fetch and optionally L2-normalize vectors
        vectors = {}
        for uid in user_ids:
            vec = raw_counts.get(uid, {})
            vectors[uid] = self.l2_normalize_sparse(vec) if l2_normalize else vec

        for i in range(len(user_ids)):
            vec1 = vectors[user_ids[i]]

            for j in range(i + 1, len(user_ids)):
                vec2 = vectors[user_ids[j]]

                cosine_dist = self.cosine_distance_sparse(vec1, vec2)
                euclidean_dist = self.euclidean_distance_sparse(vec1, vec2)

                cosine_distances.append(cosine_dist)
                euclidean_distances.append(euclidean_dist)

        return cosine_distances, euclidean_distances

    def _collect_distances_for_all_algorithms(self, user_ids, description,
                                               scale_alg_2_3: bool = True):
        """
        Helper function to collect pairwise distances for ALL algorithms (2, 3, 11, 21).
        Computes distances for both raw and L2-normalized embeddings.

        For algorithms 2 and 3, when scale_alg_2_3 is True, embeddings are first
        min-max scaled to [0, 1] per dimension before L2 normalization.

        Args:
            user_ids: List of user IDs.
            description: Description string for print output.
            scale_alg_2_3: If True, apply [0,1] min-max scaling to algorithms 2 and 3
                           before L2 normalization.

        Returns:
            (raw_cosine, raw_euclidean, l2_cosine, l2_euclidean) dictionaries.
        """
        print(f"\n  Computing pairwise distances for {description}...")
        print(f"  Users: {len(user_ids)}, Pairs per algorithm: {len(user_ids) * (len(user_ids) - 1) // 2}")
        if scale_alg_2_3:
            print(f"  [0,1] scaling enabled for Algorithms 2 and 3")

        raw_cosine = {}
        raw_euclidean = {}
        l2_cosine = {}
        l2_euclidean = {}

        # Collect distances for tensor-based algorithms (2, 3, 11)
        for alg_id in self.TENSOR_ALGORITHM_IDS:
            MAT_E_raw = self.embeddings_by_alg[alg_id]

            # Optionally scale learned embeddings before L2 normalization
            if scale_alg_2_3 and alg_id in (2, 3):
                MAT_E_scaled = self.scale_to_unit_range(MAT_E_raw)
                MAT_E_l2 = self.l2_normalize_tensor(MAT_E_scaled)
            else:
                MAT_E_l2 = self.l2_normalize_tensor(MAT_E_raw)

            shape = tuple(MAT_E_raw.shape)
            key = f"Alg_{alg_id} {shape}"

            # Raw distances
            cos_raw, euc_raw = self.compute_pairwise_distances_for_user_group(user_ids, MAT_E_raw)
            raw_cosine[key] = cos_raw
            raw_euclidean[key] = euc_raw

            # L2-normalized distances (with optional [0,1] pre-scaling for alg 2, 3)
            cos_l2, euc_l2 = self.compute_pairwise_distances_for_user_group(user_ids, MAT_E_l2)
            l2_cosine[key] = cos_l2
            l2_euclidean[key] = euc_l2

        # Collect distances for raw tool counts (algorithm 21)
        key_raw = f"Alg_{self.RAW_ALG_ID} {self._raw_tensor_shape}"

        # Raw sparse distances
        cos_sparse_raw, euc_sparse_raw = self.compute_pairwise_distances_for_user_group_raw(
            user_ids, l2_normalize=False
        )
        raw_cosine[key_raw] = cos_sparse_raw
        raw_euclidean[key_raw] = euc_sparse_raw

        # L2-normalized sparse distances
        cos_sparse_l2, euc_sparse_l2 = self.compute_pairwise_distances_for_user_group_raw(
            user_ids, l2_normalize=True
        )
        l2_cosine[key_raw] = cos_sparse_l2
        l2_euclidean[key_raw] = euc_sparse_l2

        return raw_cosine, raw_euclidean, l2_cosine, l2_euclidean

    # ==================== STATISTICS ====================
    @staticmethod
    def compute_combined_stats(raw_datasets: Dict[str, List[float]],
                               l2_datasets: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for both raw and L2-normalized distances.

        Returns:
            Dictionary mapping algorithm name to
            {"raw_min", "raw_max", "l2_min", "l2_max", "l2_mean", "l2_std", "n_samples"}.
        """
        stats = {}
        for name in l2_datasets:
            raw_arr = np.array(raw_datasets[name])
            l2_arr = np.array(l2_datasets[name])
            stats[name] = {
                "raw_min": float(np.min(raw_arr)),
                "raw_max": float(np.max(raw_arr)),
                "l2_min": float(np.min(l2_arr)),
                "l2_max": float(np.max(l2_arr)),
                "l2_mean": float(np.mean(l2_arr)),
                "l2_std": float(np.std(l2_arr, ddof=0)),
                "n_samples": len(l2_arr)
            }
        return stats

    def print_stats(self, raw_datasets: Dict[str, List[float]],
                    l2_datasets: Dict[str, List[float]],
                    title: str) -> Dict[str, Dict[str, float]]:
        """
        Compute and print distance statistics for each algorithm,
        showing both raw and L2-normalized min/max.

        Args:
            raw_datasets: Dictionary mapping algorithm name to list of raw distance values.
            l2_datasets: Dictionary mapping algorithm name to list of L2-normalized distance values.
            title: Title for the printed output section.

        Returns:
            The computed statistics dictionary.
        """
        stats = self.compute_combined_stats(raw_datasets, l2_datasets)

        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

        for name, stat in stats.items():
            print(f"\n{name}:")
            print(f"  Raw Min: {self.format_value(stat['raw_min']):>12}, "
                  f"Raw Max: {self.format_value(stat['raw_max']):>12}")
            print(f"  L2 Min:  {self.format_value(stat['l2_min']):>12}, "
                  f"L2 Max:  {self.format_value(stat['l2_max']):>12}")
            print(f"  L2 Mean: {self.format_value(stat['l2_mean']):>12}, "
                  f"L2 SD:   {self.format_value(stat['l2_std']):>12}, "
                  f"N: {stat['n_samples']}")

        print("=" * 80)

        return stats

    # ==================== CANARY 1 ANALYSIS ====================
    def compute_distances_canary_1(self, scale_alg_2_3: bool = True):
        """
        Compute distance statistics for pairwise distances within Canary 1 group
        using L2-normalized embeddings, with raw min/max for reference.
        """
        canary_users = self.get_canary_user_ids(1)

        if len(canary_users) < 2:
            print("Error: Not enough canary users in category 1.")
            return

        print("\n" + "=" * 80)
        print("L2-NORMALIZED DISTANCES WITHIN CANARY 1 GROUP (Algorithms 2, 3, 11, 21)")
        print("=" * 80)

        raw_cos, raw_euc, l2_cos, l2_euc = self._collect_distances_for_all_algorithms(
            canary_users, "Canary 1 group", scale_alg_2_3=scale_alg_2_3
        )

        self.print_stats(raw_cos, l2_cos, "COSINE DISTANCES - CANARY 1")
        self.print_stats(raw_euc, l2_euc, "EUCLIDEAN DISTANCES - CANARY 1")

    # ==================== CANARY 2 ANALYSIS ====================
    def compute_distances_canary_2(self, scale_alg_2_3: bool = True):
        """
        Compute distance statistics for pairwise distances within Canary 2 group
        using L2-normalized embeddings, with raw min/max for reference.
        """
        canary_users = self.get_canary_user_ids(2)

        if len(canary_users) < 2:
            print("Error: Not enough canary users in category 2.")
            return

        print("\n" + "=" * 80)
        print("L2-NORMALIZED DISTANCES WITHIN CANARY 2 GROUP (Algorithms 2, 3, 11, 21)")
        print("=" * 80)

        raw_cos, raw_euc, l2_cos, l2_euc = self._collect_distances_for_all_algorithms(
            canary_users, "Canary 2 group", scale_alg_2_3=scale_alg_2_3
        )

        self.print_stats(raw_cos, l2_cos, "COSINE DISTANCES - CANARY 2")
        self.print_stats(raw_euc, l2_euc, "EUCLIDEAN DISTANCES - CANARY 2")

    # ==================== ALL USERS ANALYSIS ====================
    def compute_distances_all_users(self, scale_alg_2_3: bool = True):
        """
        Compute distance statistics for pairwise distances across all users
        using L2-normalized embeddings, with raw min/max for reference.
        """
        all_users = self.all_user_ids

        if len(all_users) < 2:
            print("Error: Not enough users found in database.")
            return

        print("\n" + "=" * 80)
        print("L2-NORMALIZED DISTANCES FOR ALL USERS (Algorithms 2, 3, 11, 21)")
        print("=" * 80)

        raw_cos, raw_euc, l2_cos, l2_euc = self._collect_distances_for_all_algorithms(
            all_users, "all users", scale_alg_2_3=scale_alg_2_3
        )

        self.print_stats(raw_cos, l2_cos, "COSINE DISTANCES - ALL USERS")
        self.print_stats(raw_euc, l2_euc, "EUCLIDEAN DISTANCES - ALL USERS")

    # ==================== MANUAL TEST FUNCTIONS ====================
    @staticmethod
    def compute_distances_for_manual_embeddings(embeddings: List[List[float]]):
        """
        Takes a list of 8-dimensional embeddings, computes all pairwise cosine and
        euclidean distances for both raw and L2-normalized embeddings, and prints
        mean, SD, min and max.

        Args:
            embeddings: List of embeddings, each a list of 8 floats.
        """
        MAT_E = torch.tensor(embeddings, dtype=torch.float32)

        # ---- Raw Embeddings ----
        print("\n" + "-" * 60)
        print(f"  Input: {MAT_E.shape[0]} vectors of dimension {MAT_E.shape[1]}")
        print("-" * 60)

        print("\n  Raw embeddings:")
        for i, emb in enumerate(embeddings):
            raw_norm = torch.norm(MAT_E[i], p=2).item()
            print(f"    [{i}] {emb}  (norm={raw_norm:.6f})")

        # ---- Raw Pairwise Distances ----
        raw_cosine = []
        raw_euclidean = []

        print("\n" + "-" * 60)
        print("  PAIRWISE DISTANCES (RAW EMBEDDINGS)")
        print("-" * 60)

        for i in range(MAT_E.shape[0]):
            for j in range(i + 1, MAT_E.shape[0]):
                cos_dist = CDistanceFunctions.cosine_distance_tensors(MAT_E[i], MAT_E[j]).item()
                euc_dist = CDistanceFunctions.euclidean_distance_tensors(MAT_E[i], MAT_E[j]).item()
                raw_cosine.append(cos_dist)
                raw_euclidean.append(euc_dist)
                print(f"    Pair ({i},{j}): Cosine={cos_dist:.6f}, Euclidean={euc_dist:.6f}")

        raw_cos_arr = np.array(raw_cosine)
        raw_euc_arr = np.array(raw_euclidean)

        print(f"\n  Cosine Distances ({len(raw_cosine)} pairs):")
        print(f"    Min:  {np.min(raw_cos_arr):.6f}, Max:  {np.max(raw_cos_arr):.6f}")
        print(f"    Mean: {np.mean(raw_cos_arr):.6f}, SD:   {np.std(raw_cos_arr, ddof=0):.6f}")

        print(f"\n  Euclidean Distances ({len(raw_euclidean)} pairs):")
        print(f"    Min:  {np.min(raw_euc_arr):.6f}, Max:  {np.max(raw_euc_arr):.6f}")
        print(f"    Mean: {np.mean(raw_euc_arr):.6f}, SD:   {np.std(raw_euc_arr, ddof=0):.6f}")

        # ---- L2-Normalized Embeddings ----
        norms = torch.norm(MAT_E, p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-12)
        MAT_E_norm = MAT_E / norms

        print("\n" + "-" * 60)
        print("  L2-NORMALIZED EMBEDDINGS")
        print("-" * 60)

        for i in range(MAT_E_norm.shape[0]):
            vals = [f"{v:.6f}" for v in MAT_E_norm[i].tolist()]
            norm_val = torch.norm(MAT_E_norm[i], p=2).item()
            print(f"    [{i}] [{', '.join(vals)}]  (norm={norm_val:.6f})")

        # ---- L2-Normalized Pairwise Distances ----
        l2_cosine = []
        l2_euclidean = []

        print("\n" + "-" * 60)
        print("  PAIRWISE DISTANCES (L2-NORMALIZED EMBEDDINGS)")
        print("-" * 60)

        for i in range(MAT_E_norm.shape[0]):
            for j in range(i + 1, MAT_E_norm.shape[0]):
                cos_dist = CDistanceFunctions.cosine_distance_tensors(MAT_E_norm[i], MAT_E_norm[j]).item()
                euc_dist = CDistanceFunctions.euclidean_distance_tensors(MAT_E_norm[i], MAT_E_norm[j]).item()
                l2_cosine.append(cos_dist)
                l2_euclidean.append(euc_dist)
                print(f"    Pair ({i},{j}): Cosine={cos_dist:.6f}, Euclidean={euc_dist:.6f}")

        l2_cos_arr = np.array(l2_cosine)
        l2_euc_arr = np.array(l2_euclidean)

        print(f"\n  Cosine Distances ({len(l2_cosine)} pairs):")
        print(f"    Min:  {np.min(l2_cos_arr):.6f}, Max:  {np.max(l2_cos_arr):.6f}")
        print(f"    Mean: {np.mean(l2_cos_arr):.6f}, SD:   {np.std(l2_cos_arr, ddof=0):.6f}")

        print(f"\n  Euclidean Distances ({len(l2_euclidean)} pairs):")
        print(f"    Min:  {np.min(l2_euc_arr):.6f}, Max:  {np.max(l2_euc_arr):.6f}")
        print(f"    Mean: {np.mean(l2_euc_arr):.6f}, SD:   {np.std(l2_euc_arr, ddof=0):.6f}")

        print("\n" + "-" * 60)

    @staticmethod
    def test_with_manual_values():
        """
        Test with 4 hand-crafted 8-dimensional embeddings with different value ranges.
        """
        print("\n" + "=" * 80)
        print("MANUAL TEST: L2-NORMALIZED DISTANCES FOR HAND-CRAFTED EMBEDDINGS")
        print("=" * 80)

        embeddings = [
            # Small values, close to zero
            [0.01, 0.02, 0.03, 0.01, 0.02, 0.01, 0.03, 0.02],
            # Medium values
            [1.0, 2.0, 3.0, 1.5, 2.5, 1.0, 3.5, 2.0],
            # Large values
            [100.0, 200.0, 150.0, 300.0, 250.0, 100.0, 350.0, 200.0],
            # Mixed range with negative values
            [-5.0, 10.0, -3.0, 8.0, -1.0, 6.0, -4.0, 12.0],
        ]

        CDistanceAnalysis_Baselines_L2.compute_distances_for_manual_embeddings(embeddings)

    # ==================== MAIN ANALYSIS FUNCTION ====================
    def print_all_baseline_analysis(self, include_all_users: bool = False,
                                     scale_alg_2_3: bool = True):
        """
        Print comprehensive distance analysis using L2-normalized embeddings:
        - Distances within Canary 1 group
        - Distances within Canary 2 group
        - Distances for all users (controlled by include_all_users flag)

        All embeddings are L2-normalized to unit length before computing distances.
        For algorithms 2 and 3, embeddings are optionally min-max scaled to [0, 1]
        per dimension before L2 normalization.
        Distance values are reported as-is (no min-max scaling on distances).

        Args:
            include_all_users: If True, compute distances for all users
                              (can be very slow for large datasets)
            scale_alg_2_3: If True (default), apply [0,1] min-max scaling to
                          algorithms 2 and 3 embeddings before L2 normalization.
        """
        print("\n" + "=" * 80)
        print("   L2-NORMALIZED EMBEDDING BASELINE ANALYSIS (Algorithms 2, 3, 11, 21)")
        if scale_alg_2_3:
            print("   [0,1] scaling enabled for Algorithms 2 and 3")
        print("=" * 80 + "\n")

        # ---- CANARY 1 ----
        self.compute_distances_canary_1(scale_alg_2_3=scale_alg_2_3)

        # ---- CANARY 2 ----
        self.compute_distances_canary_2(scale_alg_2_3=scale_alg_2_3)

        # ---- ALL USERS (optional, slow) ----
        if include_all_users:
            self.compute_distances_all_users(scale_alg_2_3=scale_alg_2_3)
        else:
            print("\n  [Skipping all users analysis - set include_all_users=True to enable]")


if __name__ == "__main__":
    # Run manual test first (no database needed)
    #CDistanceAnalysis_Baselines_L2.test_with_manual_values()

    # Uncomment below to run full analysis (requires database)
    print("\nRunning L2-Normalized Embedding Baseline Analysis (All 4 Algorithms)...")
    analysis = CDistanceAnalysis_Baselines_L2()
    analysis.print_all_baseline_analysis(include_all_users=True)
