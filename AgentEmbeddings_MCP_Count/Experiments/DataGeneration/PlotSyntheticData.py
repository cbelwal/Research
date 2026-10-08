import os,sys
# ----------------------------------------------
# Ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.Plots.CPlotSyntheticData import CPlotSyntheticData
from Experiments.CConfig import CConfig

if __name__ == "__main__":
    output_directory = os.path.join(
        topRootPath,
        "Experiments",
        "Data",
        "ExperimentResults",
        f"a{CConfig.MAX_AGENTS}",
    )
    plotter = CPlotSyntheticData(outputDirectory=output_directory)
    print("Plotting synthetic data...")
    plotter.generate_all_plots()
    print("Plots generated and saved successfully.")
