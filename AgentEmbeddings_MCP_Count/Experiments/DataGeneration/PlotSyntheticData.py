import os,sys
# ----------------------------------------------
# Ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.Plots.CPlotSyntheticData import CPlotSyntheticData

if __name__ == "__main__":
    plotter = CPlotSyntheticData()
    print("Plotting synthetic data...")
    plotter.generate_all_plots()
    print("Plots generated and saved successfully.")

