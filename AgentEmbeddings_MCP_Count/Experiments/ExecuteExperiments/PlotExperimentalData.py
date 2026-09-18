import os,sys

# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)

#----------------------------------------------
from Experiments.Plots.CPlotExperimentalData import CPlotExperimentalData

def generate_plots_for_experiment_data(algID:int):
    plotter = CPlotExperimentalData(algID=algID)
    plotter.generate_all_plots()
    
if __name__ == "__main__":
    for algID in [2, 3]:
        print(f"*** Generating plots for Algorithm {algID}...")
        generate_plots_for_experiment_data(algID=algID)
        
