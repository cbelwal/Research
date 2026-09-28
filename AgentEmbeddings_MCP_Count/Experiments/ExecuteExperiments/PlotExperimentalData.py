import os,sys
import argparse

# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)

#----------------------------------------------
from Experiments.Plots.CPlotExperimentalData import CPlotExperimentalData
from Experiments.CConfig import CConfig

def generate_plots_for_experiment_data(algID:int, embeddingDimensions:int):
    plotter = CPlotExperimentalData(
        algID=algID,
        embeddingDimensions=embeddingDimensions,
    )
    plotter.generate_all_plots()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=list(CConfig.EMBEDDING_DIMENSIONS_TO_RUN),
    )
    parser.add_argument(
        "--algorithms",
        type=int,
        nargs="+",
        default=[2, 3],
        choices=[2, 3],
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    for embedding_dimensions in args.dimensions:
        for algID in args.algorithms:
            print(
                f"*** Generating plots for Algorithm {algID}, "
                f"dimension {embedding_dimensions}..."
            )
            generate_plots_for_experiment_data(
                algID=algID,
                embeddingDimensions=embedding_dimensions,
            )
        
