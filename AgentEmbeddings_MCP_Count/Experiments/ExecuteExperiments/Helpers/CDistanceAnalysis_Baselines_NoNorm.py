# Computes raw (unnormalized) Cosine and Euclidean distances for embeddings from:
# Algorithms 2, 3, PCA Baseline (11), and Raw Tool Counts (21)
# No normalization is applied to embeddings or distances.
# Reports mean, SD, min and max of the raw distance values.
#----------------------
# There is no normalization here. 
# 
# For algorithms 2 and 3, we also compute optional [0,1] min-max scaling before computing distances,
# to show how much the scale of the original embeddings affects the distance 
# distributions. For algorithm 11 (PCA baseline) and 21 (raw tool counts), 
# we only compute raw distances since they are already on a different scale.
#
# This is not used in the main paper, and is used only for comparison purposes.

import math
import os
import torch
import numpy as np
from typing import Dict, List
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

# No normalization of embeddings or distances
class CDistanceAnalysis_Baselines_NoNorm:
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
        Returns 1 - cosine_similarity.
        """
        all_keys = set(vec1_dict.keys()) | set(vec2_dict.keys())

        dot_product = 0.0
        magnitude_1 = 0.0
        magnitude_2 = 0.0

        for key in all_keys:
            val1 = vec1_dict.get(key, 0)
            val2 = vec2_dict.get(key, 0)
            dot_product += val1 * val2
            magnitude_1 += val1 * val1
            magnitude_2 += val2 * val2

        magnitude_1 = math.sqrt(magnitude_1)
        magnitude_2 = math.sqrt(magnitude_2)

        if magnitude_1 == 0 or magnitude_2 == 0:
            return 1.0  # Maximum distance if one vector is zero

        cosine_similarity = dot_product / (magnitude_1 * magnitude_2)
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

    def compute_pairwise_distances_for_user_group_raw(self, user_ids):
        """
        Compute all pairwise cosine and euclidean distances within a group of users
        using raw tool counts (sparse dictionary format).
        Returns lists of cosine_distances and euclidean_distances.
        """
        raw_counts = self.get_raw_tool_counts()
        cosine_distances = []
        euclidean_distances = []

        for i in range(len(user_ids)):
            user_id_1 = user_ids[i]
            vec1 = raw_counts.get(user_id_1, {})

            for j in range(i + 1, len(user_ids)):
                user_id_2 = user_ids[j]
                vec2 = raw_counts.get(user_id_2, {})

                cosine_dist = self.cosine_distance_sparse(vec1, vec2)
                euclidean_dist = self.euclidean_distance_sparse(vec1, vec2)

                cosine_distances.append(cosine_dist)
                euclidean_distances.append(euclidean_dist)

        return cosine_distances, euclidean_distances

    def _collect_distances_for_all_algorithms(self, user_ids, description,
                                               scale_alg_2_3: bool = True):
        """
        Helper function to collect pairwise distances for ALL algorithms (2, 3, 11, 21).
        Always computes raw (unscaled) distances. When scale_alg_2_3 is True, also
        computes distances on [0,1]-scaled embeddings for algorithms 2 and 3.

        Args:
            user_ids: List of user IDs.
            description: Description string for print output.
            scale_alg_2_3: If True, also compute distances with [0,1] min-max scaling
                           for algorithms 2 and 3.

        Returns:
            (raw_cosine, raw_euclidean, scaled_cosine, scaled_euclidean) dictionaries.
            For algorithms 11 and 21 (and 2/3 when scale_alg_2_3=False),
            scaled values are the same as raw values.
        """
        print(f"\n  Computing pairwise distances for {description}...")
        print(f"  Users: {len(user_ids)}, Pairs per algorithm: {len(user_ids) * (len(user_ids) - 1) // 2}")
        if scale_alg_2_3:
            print(f"  [0,1] scaling enabled for Algorithms 2 and 3")

        raw_cosine = {}
        raw_euclidean = {}
        scaled_cosine = {}
        scaled_euclidean = {}

        # Collect distances for tensor-based algorithms (2, 3, 11)
        for alg_id in self.TENSOR_ALGORITHM_IDS:
            MAT_E_raw = self.embeddings_by_alg[alg_id]
            shape = tuple(MAT_E_raw.shape)
            key = f"Alg_{alg_id} {shape}"

            # Raw distances (no scaling)
            cos_raw, euc_raw = self.compute_pairwise_distances_for_user_group(user_ids, MAT_E_raw)
            raw_cosine[key] = cos_raw
            raw_euclidean[key] = euc_raw

            # Scaled distances for learned embedding algorithms
            if scale_alg_2_3 and alg_id in (2, 3):
                MAT_E_scaled = self.scale_to_unit_range(MAT_E_raw)
                cos_scaled, euc_scaled = self.compute_pairwise_distances_for_user_group(
                    user_ids, MAT_E_scaled
                )
                scaled_cosine[key] = cos_scaled
                scaled_euclidean[key] = euc_scaled
            else:
                scaled_cosine[key] = cos_raw
                scaled_euclidean[key] = euc_raw

        # Collect distances for raw tool counts (algorithm 21)
        cos_sparse, euc_sparse = self.compute_pairwise_distances_for_user_group_raw(user_ids)
        key_raw = f"Alg_{self.RAW_ALG_ID} {self._raw_tensor_shape}"
        raw_cosine[key_raw] = cos_sparse
        raw_euclidean[key_raw] = euc_sparse
        scaled_cosine[key_raw] = cos_sparse
        scaled_euclidean[key_raw] = euc_sparse

        return raw_cosine, raw_euclidean, scaled_cosine, scaled_euclidean

    # ==================== STATISTICS ====================
    @staticmethod
    def compute_combined_stats(raw_datasets: Dict[str, List[float]],
                               scaled_datasets: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for both raw and [0,1]-scaled distances.

        Returns:
            Dictionary mapping algorithm name to
            {"raw_min", "raw_max", "raw_mean", "raw_std",
             "scaled_min", "scaled_max", "scaled_mean", "scaled_std", "n_samples"}.
        """
        stats = {}
        for name in raw_datasets:
            raw_arr = np.array(raw_datasets[name])
            scaled_arr = np.array(scaled_datasets[name])
            stats[name] = {
                "raw_min": float(np.min(raw_arr)),
                "raw_max": float(np.max(raw_arr)),
                "raw_mean": float(np.mean(raw_arr)),
                "raw_std": float(np.std(raw_arr, ddof=0)),
                "scaled_min": float(np.min(scaled_arr)),
                "scaled_max": float(np.max(scaled_arr)),
                "scaled_mean": float(np.mean(scaled_arr)),
                "scaled_std": float(np.std(scaled_arr, ddof=0)),
                "n_samples": len(raw_arr)
            }
        return stats

    def print_stats(self, raw_datasets: Dict[str, List[float]],
                    scaled_datasets: Dict[str, List[float]],
                    title: str) -> Dict[str, Dict[str, float]]:
        """
        Compute and print distance statistics for each algorithm,
        showing both raw and [0,1]-scaled min/max/mean/SD.

        Args:
            raw_datasets: Dictionary mapping algorithm name to list of raw distance values.
            scaled_datasets: Dictionary mapping algorithm name to list of [0,1]-scaled distance values.
            title: Title for the printed output section.

        Returns:
            The computed statistics dictionary.
        """
        stats = self.compute_combined_stats(raw_datasets, scaled_datasets)

        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

        for name, stat in stats.items():
            # Detect if this is an algorithm with [0,1] scaling (Alg 2 or 3)
            is_scaled = name.startswith("Alg_2 ") or name.startswith("Alg_3 ")

            if is_scaled:
                print(f"\n{name}:")
                print(f"  Raw         Min:  {self.format_value(stat['raw_min']):>12}, "
                      f"Max:  {self.format_value(stat['raw_max']):>12}")
                print(f"  Raw         Mean: {self.format_value(stat['raw_mean']):>12}, "
                      f"SD:   {self.format_value(stat['raw_std']):>12}")
                print(f"  Scaled[0,1] Min:  {self.format_value(stat['scaled_min']):>12}, "
                      f"Max:  {self.format_value(stat['scaled_max']):>12}")
                print(f"  Scaled[0,1] Mean: {self.format_value(stat['scaled_mean']):>12}, "
                      f"SD:   {self.format_value(stat['scaled_std']):>12}, "
                      f"N: {stat['n_samples']}")
            else:
                print(f"\n{name}:  (no scaling)")
                print(f"  Min:  {self.format_value(stat['raw_min']):>12}, "
                      f"Max:  {self.format_value(stat['raw_max']):>12}")
                print(f"  Mean: {self.format_value(stat['raw_mean']):>12}, "
                      f"SD:   {self.format_value(stat['raw_std']):>12}, "
                      f"N: {stat['n_samples']}")

        print("=" * 80)

        return stats

    # ==================== CANARY 1 ANALYSIS ====================
    def compute_distances_canary_1(self, scale_alg_2_3: bool = True):
        """
        Compute raw distance statistics for pairwise distances within Canary 1 group.
        """
        canary_users = self.get_canary_user_ids(1)

        if len(canary_users) < 2:
            print("Error: Not enough canary users in category 1.")
            return

        print("\n" + "=" * 80)
        print("RAW DISTANCES WITHIN CANARY 1 GROUP (Algorithms 2, 3, 11, 21)")
        print("=" * 80)

        raw_cos, raw_euc, scaled_cos, scaled_euc = self._collect_distances_for_all_algorithms(
            canary_users, "Canary 1 group", scale_alg_2_3=scale_alg_2_3
        )

        self.print_stats(raw_cos, scaled_cos, "COSINE DISTANCES - CANARY 1")
        self.print_stats(raw_euc, scaled_euc, "EUCLIDEAN DISTANCES - CANARY 1")

    # ==================== CANARY 2 ANALYSIS ====================
    def compute_distances_canary_2(self, scale_alg_2_3: bool = True):
        """
        Compute raw distance statistics for pairwise distances within Canary 2 group.
        """
        canary_users = self.get_canary_user_ids(2)

        if len(canary_users) < 2:
            print("Error: Not enough canary users in category 2.")
            return

        print("\n" + "=" * 80)
        print("RAW DISTANCES WITHIN CANARY 2 GROUP (Algorithms 2, 3, 11, 21)")
        print("=" * 80)

        raw_cos, raw_euc, scaled_cos, scaled_euc = self._collect_distances_for_all_algorithms(
            canary_users, "Canary 2 group", scale_alg_2_3=scale_alg_2_3
        )

        self.print_stats(raw_cos, scaled_cos, "COSINE DISTANCES - CANARY 2")
        self.print_stats(raw_euc, scaled_euc, "EUCLIDEAN DISTANCES - CANARY 2")

    # ==================== ALL USERS ANALYSIS ====================
    def compute_distances_all_users(self, scale_alg_2_3: bool = True):
        """
        Compute raw distance statistics for pairwise distances across all users.
        """
        all_users = self.all_user_ids

        if len(all_users) < 2:
            print("Error: Not enough users found in database.")
            return

        print("\n" + "=" * 80)
        print("RAW DISTANCES FOR ALL USERS (Algorithms 2, 3, 11, 21)")
        print("=" * 80)

        raw_cos, raw_euc, scaled_cos, scaled_euc = self._collect_distances_for_all_algorithms(
            all_users, "all users", scale_alg_2_3=scale_alg_2_3
        )

        self.print_stats(raw_cos, scaled_cos, "COSINE DISTANCES - ALL USERS")
        self.print_stats(raw_euc, scaled_euc, "EUCLIDEAN DISTANCES - ALL USERS")

    # ==================== MAIN ANALYSIS FUNCTION ====================
    def print_all_baseline_analysis(self, include_all_users: bool = False,
                                     scale_alg_2_3: bool = True):
        """
        Print comprehensive raw (unnormalized) distance baseline analysis including:
        - Raw distances within Canary 1 group
        - Raw distances within Canary 2 group
        - Raw distances for all users (controlled by include_all_users flag)

        No normalization is applied to embeddings or distances.
        For algorithms 2 and 3, embeddings are optionally min-max scaled to [0, 1]
        per dimension before computing distances.

        Args:
            include_all_users: If True, compute distances for all users
                              (can be very slow for large datasets)
            scale_alg_2_3: If True (default), apply [0,1] min-max scaling to
                          algorithms 2 and 3 embeddings before computing distances.
        """
        print("\n" + "=" * 80)
        print("   RAW (UNNORMALIZED) EMBEDDING BASELINE ANALYSIS (Algorithms 2, 3, 11, 21)")
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
    print("Running Raw (Unnormalized) Embedding Baseline Analysis (All 4 Algorithms)...")
    analysis = CDistanceAnalysis_Baselines_NoNorm()
    # Set include_all_users=True to compute for all users (slow)
    analysis.print_all_baseline_analysis(include_all_users=False)
