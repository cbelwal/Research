import os, sys

# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)

#----------------------------------------------
from Experiments.Plots.CPlotBaselineClustering import CPlotBaselineClustering

def generate_baseline_clustering_plots():
    plotter = CPlotBaselineClustering()
    plotter.generate_all_plots()

if __name__ == "__main__":
    print("Generating Baseline Clustering Plots...")
    generate_baseline_clustering_plots()
