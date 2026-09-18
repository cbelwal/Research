"""
Plotting silhouette scores across all algorithms (2, 3, 11, 21).
"""
import os, sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.Plots.CPlotCommon import CPlotCommon
from Experiments.ExecuteExperiments.Helpers.CClusteringAnalysis_Baselines_All import CClusteringAnalysis_Baselines_All

class CPlotBaselineClustering:
    ALG_NAMES = {
        2: "Algorithm 2",
        3: "Algorithm 3",
        11: "PCA Baseline",
        21: "Raw Tool Counts",
    }

    def __init__(self):
        self.clustering = CClusteringAnalysis_Baselines_All()

    def plot_silhouette_scores_for_range(self, min_clusters=2, max_clusters=10, saveFile=False):
        """
        Collect silhouette scores for all algorithms across a range of cluster counts
        and plot them as a multi-line chart.
        """
        cluster_range = list(range(min_clusters, max_clusters + 1))

        y_series = {}
        for alg_id in self.clustering.ALGORITHM_IDS:
            scores = []
            for num_clusters in cluster_range:
                score = self.clustering.compute_silhouette_score(alg_id, num_clusters)
                scores.append(score if score is not None else 0.0)
            y_series[self.ALG_NAMES[alg_id]] = scores

        CPlotCommon.plot_multi_line_xy(
            x=cluster_range,
            y_series=y_series,
            title="Silhouette Scores vs Number of Clusters - All Algorithms",
            xlabel="Number of Clusters",
            ylabel="Silhouette Score",
            saveFile=saveFile
        )

    def generate_all_plots(self):
        self.plot_silhouette_scores_for_range(saveFile=True)


if __name__ == "__main__":
    plotter = CPlotBaselineClustering()
    plotter.generate_all_plots()
