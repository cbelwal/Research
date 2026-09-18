# Given a user ID and K, find the K closest users from each algorithm
import os,sys
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

class CTopKUsers:
    ALGORITHM_IDS = [2, 3, 11, 21]  # All active algorithms including baselines

    def __init__(self):
        self.dbManager = CDatabaseManager()
        self.canary_users = self.dbManager.get_canary_users()
        self.all_user_ids = self.dbManager.get_all_user_ids()
        # Load embeddings for each algorithm
        self.embeddings_by_alg = {}
        for alg_id in self.ALGORITHM_IDS:
            store = CResultsStore(algID=alg_id)
            self.embeddings_by_alg[alg_id] = store.load_embeddings()

    def get_non_canary_user_ids(self):
        """Return all user IDs that are not part of canary 1 or canary 2 groups."""
        canary_user_set = set()
        for canary_id in self.canary_users:
            canary_user_set.update(self.canary_users[canary_id])
        non_canary_ids = [uid for uid in self.all_user_ids if uid not in canary_user_set]
        return non_canary_ids

    @staticmethod
    def format_distance(value):
        """Format distance value to handle very small numbers."""
        if value == 0:
            return "0.000000"
        elif abs(value) < 0.000001:
            return f"{value:.6e}"
        else:
            return f"{value:.6f}"

    def _compute_top_k_for_all_algorithms(self, user_id, k):
        """
        Compute top K closest users for the given user_id across all algorithms.
        Returns a dict: { alg_id: { 'cosine': [(uid, dist), ...], 'euclidean': [(uid, dist), ...] } }
        """
        results = {}
        for alg_id in self.ALGORITHM_IDS:
            MAT_E = self.embeddings_by_alg[alg_id]
            # CAUTION: MAT_E is 0-indexed, user IDs are 1-indexed in DB
            target_embedding = MAT_E[user_id - 1]

            cosine_distances = []
            euclidean_distances = []
            for other_id in self.all_user_ids:
                if other_id == user_id:
                    continue
                other_embedding = MAT_E[other_id - 1]
                cosine_dist = CDistanceFunctions.cosine_distance_tensors(
                    target_embedding, other_embedding
                ).item()
                euclidean_dist = CDistanceFunctions.euclidean_distance_tensors(
                    target_embedding, other_embedding
                ).item()
                cosine_distances.append((other_id, cosine_dist))
                euclidean_distances.append((other_id, euclidean_dist))

            cosine_distances.sort(key=lambda x: x[1], reverse=True)
            euclidean_distances.sort(key=lambda x: x[1], reverse=True)

            results[alg_id] = {
                'cosine': cosine_distances[:k],
                'euclidean': euclidean_distances[:k]
            }
        return results

    def find_top_k_closest_users(self, user_id, k):
        """
        For the given user_id, find the K closest users from each algorithm.
        Top K is computed separately by cosine distance and euclidean distance.
        """
        if user_id not in self.all_user_ids:
            print(f"Error: User ID {user_id} not found in database.")
            return None

        results = self._compute_top_k_for_all_algorithms(user_id, k)

        print("\n" + "=" * 70)
        print(f"TOP {k} CLOSEST USERS TO USER {user_id}")
        print("=" * 70)

        for alg_id in self.ALGORITHM_IDS:
            top_k_cosine = results[alg_id]['cosine']
            top_k_euclidean = results[alg_id]['euclidean']

            shape = tuple(self.embeddings_by_alg[alg_id].shape)
            print(f"\nAlgorithm {alg_id:2d} (tensor shape: {shape}):")
            print(f"  Top {k} by Cosine Distance:    {[uid for uid, _ in top_k_cosine]}")
            for uid, dist in top_k_cosine:
                print(f"    User {uid:5d} -> {self.format_distance(dist)}")
            print(f"    LaTeX: {' & '.join(str(uid) for uid, _ in top_k_cosine)}")
            print(f"  Top {k} by Euclidean Distance: {[uid for uid, _ in top_k_euclidean]}")
            for uid, dist in top_k_euclidean:
                print(f"    User {uid:5d} -> {self.format_distance(dist)}")
            print(f"    LaTeX: {' & '.join(str(uid) for uid, _ in top_k_euclidean)}")

        print("=" * 70)
        return results

    def find_common_top_k_between_alg_2_and_3(self, user_id, k):
        """
        Find and print common top K users between algorithms 2 and 3,
        separately for cosine and euclidean distances.
        """
        if user_id not in self.all_user_ids:
            print(f"Error: User ID {user_id} not found in database.")
            return

        results = self._compute_top_k_for_all_algorithms(user_id, k)

        cosine_ids_alg2 = set(uid for uid, _ in results[2]['cosine'])
        cosine_ids_alg3 = set(uid for uid, _ in results[3]['cosine'])
        common_cosine = sorted(cosine_ids_alg2 & cosine_ids_alg3)

        euclidean_ids_alg2 = set(uid for uid, _ in results[2]['euclidean'])
        euclidean_ids_alg3 = set(uid for uid, _ in results[3]['euclidean'])
        common_euclidean = sorted(euclidean_ids_alg2 & euclidean_ids_alg3)

        print("\n" + "=" * 70)
        print(f"COMMON TOP {k} USERS BETWEEN ALGORITHM 2 AND 3 (User {user_id})")
        print("=" * 70)

        shape_2 = tuple(self.embeddings_by_alg[2].shape)
        shape_3 = tuple(self.embeddings_by_alg[3].shape)

        print(f"\n  By Cosine Distance:    ({len(common_cosine)} common) {common_cosine}")
        print(f"    Algorithm  2 (tensor shape: {shape_2}) top {k}: {[uid for uid, _ in results[2]['cosine']]}")
        print(f"    Algorithm  3 (tensor shape: {shape_3}) top {k}: {[uid for uid, _ in results[3]['cosine']]}")

        print(f"\n  By Euclidean Distance: ({len(common_euclidean)} common) {common_euclidean}")
        print(f"    Algorithm  2 (tensor shape: {shape_2}) top {k}: {[uid for uid, _ in results[2]['euclidean']]}")
        print(f"    Algorithm  3 (tensor shape: {shape_3}) top {k}: {[uid for uid, _ in results[3]['euclidean']]}")

        print("=" * 70)

    def find_common_top_k_between_all_algorithms(self, user_id, k):
        """
        Find and print common top K users across all algorithms,
        separately for cosine and euclidean distances.
        """
        if user_id not in self.all_user_ids:
            print(f"Error: User ID {user_id} not found in database.")
            return

        results = self._compute_top_k_for_all_algorithms(user_id, k)

        # Intersection across all algorithms
        cosine_sets = [set(uid for uid, _ in results[alg_id]['cosine']) for alg_id in self.ALGORITHM_IDS]
        common_cosine = sorted(cosine_sets[0].intersection(*cosine_sets[1:]))

        euclidean_sets = [set(uid for uid, _ in results[alg_id]['euclidean']) for alg_id in self.ALGORITHM_IDS]
        common_euclidean = sorted(euclidean_sets[0].intersection(*euclidean_sets[1:]))

        print("\n" + "=" * 70)
        print(f"COMMON TOP {k} USERS ACROSS ALL ALGORITHMS (User {user_id})")
        print(f"Algorithms: {self.ALGORITHM_IDS}")
        print("=" * 70)

        print(f"\n  By Cosine Distance:    ({len(common_cosine)} common) {common_cosine}")
        for alg_id in self.ALGORITHM_IDS:
            shape = tuple(self.embeddings_by_alg[alg_id].shape)
            print(f"    Algorithm {alg_id:2d} (tensor shape: {shape}) top {k}: {[uid for uid, _ in results[alg_id]['cosine']]}")

        print(f"\n  By Euclidean Distance: ({len(common_euclidean)} common) {common_euclidean}")
        for alg_id in self.ALGORITHM_IDS:
            shape = tuple(self.embeddings_by_alg[alg_id].shape)
            print(f"    Algorithm {alg_id:2d} (tensor shape: {shape}) top {k}: {[uid for uid, _ in results[alg_id]['euclidean']]}")

        print("=" * 70)


if __name__ == "__main__":
    topK = CTopKUsers()

    # Get a non-canary user to test with
    non_canary_ids = topK.get_non_canary_user_ids()
    print(f"Total non-canary users: {len(non_canary_ids)}")
    print(f"Sample non-canary user IDs: {non_canary_ids[:10]}")

    # Find top 5 closest users for the first non-canary user
    test_user_id = non_canary_ids[0]
    topK.find_top_k_closest_users(user_id=test_user_id, k=5)

    # Find common top K between algorithm 2 and 3
    topK.find_common_top_k_between_alg_2_and_3(user_id=test_user_id, k=5)

    # Find common top K across all 4 algorithms
    topK.find_common_top_k_between_all_algorithms(user_id=test_user_id, k=5)
