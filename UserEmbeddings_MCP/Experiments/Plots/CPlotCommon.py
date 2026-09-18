"""
Source: https://www.statology.org/creating-manipulating-polynomials-numpy/
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------

class CPlotCommon:
    def __init__(self):
        matplotlib.use('Agg')
        pass

    """
    Takes the title of the plot and returns the file path to 
    save the plot image
    """
    @staticmethod
    def get_plot_file_path(title) -> str:
        # Remove spaces and special characters from title for file name
        title_cleaned = title.replace(" ","_").lower()
        title_cleaned = title_cleaned.replace("-","_")
        title_cleaned = title_cleaned.replace(",","")
        title_cleaned = title_cleaned.replace(":","_")
        title_cleaned = title_cleaned.replace("/","_")
        # Remove double or triple underscores if any
        title_cleaned = title_cleaned.replace("__","_")
        title_cleaned = title_cleaned.replace("__","_") # for triples



        folderPath =    os.path.dirname(
                        os.path.dirname( #Execute Experiments
                        #os.path.dirname( #UserEmbeddings
                        os.path.abspath(__file__)))
        resultFolder = os.path.join(folderPath, "Data")
        resultFolder = os.path.join(resultFolder, "ExperimentResults")
        imageFilePath = os.path.join(resultFolder, f"{title_cleaned}.svg")
        return imageFilePath

    @staticmethod
    def plot_polynomial_fit(x, y,
                            model,
                            title="Fitted Polynomial Curve and Data Points"):
        # Generate x values for the fitted polynomial curve
        x_curve = np.linspace(min(x), max(x), 100)

        # Calculate y values for the fitted polynomial curve
        y_curve = model(x_curve)

        # Create the plot
        plt.plot(x, y, 'o', label='Data Points')
        plt.plot(x_curve, y_curve, label='Fitted Polynomial')

        # Add labels and title
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)

        # Add legend
        plt.legend()

        # Display the plot
        plt.show()

    @staticmethod
    def save_or_display_plot(plt,title,saveFile):
        if saveFile:
             # auto resize plot window before saving
            plt.tight_layout()
            filePath = CPlotCommon.get_plot_file_path(title)
            plt.savefig(filePath, format='svg')
            # add time delay to ensure file is saved before showing
            #time.sleep(2)
            #plt.show(block=False)
        plt.show()

    @staticmethod
    def plot_scatter_y(y,title="Data Points",
                       xlabel="x",
                       ylabel="y",
                       saveFile=False):
        x = np.linspace(1, len(y)-1, len(y))
        plt.scatter(x,y,marker="x") #(x, y, 'x')#, label='Data Points')

        # Add labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.title(title)

        # Add legend
        plt.legend()

        #save plot as image
        CPlotCommon.save_or_display_plot(plt,title,saveFile)
    
    """
    When both x and y are given
    """
    @staticmethod
    def plot_scatter_xy(x,
                        y,
                        title="Data Points",
                       xlabel="x",
                       ylabel="y",
                       saveFile=False):
        plt.scatter(x, y, marker='x')

        # Add labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.title(title)

        # Add legend
        plt.legend()

        #save plot as image
        CPlotCommon.save_or_display_plot(plt,title,saveFile)
    
    @staticmethod
    def plot_line_y(y,title="Data Points",
                       xlabel="x",
                       ylabel="y",
                       xstart=1,
                       saveFile=False):
        x = np.linspace(xstart, len(y)-xstart, len(y))
        plt.plot(x,y, marker='o',linestyle='-') 

        # Add labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.title(title)

        # Add legend
        plt.legend()

        #save plot as image
        CPlotCommon.save_or_display_plot(plt,title,saveFile)
    

    @staticmethod
    def plot_histogram_y(y,
                          bins=10,
                          title="Histogram",
                          xlabel="Value",
                          ylabel="Frequency",
                          saveFile=False):
        plt.hist(y, bins=bins, edgecolor='black')

        # Add labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.title(title)

        #save plot as image
        CPlotCommon.save_or_display_plot(plt,title,saveFile)

    
    @staticmethod
    def plot_bar_y(y,
                    title="Bar chart",
                    xlabel="Value",
                    ylabel="Frequency",
                    saveFile=False):
        x = np.linspace(1, len(y), len(y))
        # plot bar chart
        plt.bar(x, y, edgecolor='black',bottom=0)

        # Add labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.title(title)
        #TODO: Bar chart should start with 1
        
        #save plot as image
        CPlotCommon.save_or_display_plot(plt,title,saveFile)

    @staticmethod
    def plot_multi_line_xy(x,
                           y_series: dict,
                           title="Multi-Line Plot",
                           xlabel="x",
                           ylabel="y",
                           saveFile=False):
        """
        Plot multiple line series on the same axes with distinct markers and a legend.

        Args:
            x: Shared x-axis values (list or array).
            y_series: Dictionary mapping series name (str) to list of y values.
            title: Plot title (also used for file name when saving).
            xlabel: X-axis label.
            ylabel: Y-axis label.
            saveFile: If True, save plot as SVG.
        """
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

        plt.figure(figsize=(10, 6))

        for idx, (name, y_values) in enumerate(y_series.items()):
            marker = markers[idx % len(markers)]
            plt.plot(x, y_values, marker=marker, linestyle='-',
                     linewidth=2, markersize=7, label=name)

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(x)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)

        CPlotCommon.save_or_display_plot(plt, title, saveFile)

# For Testing
if __name__ == "__main__":
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2.3, 2.5, 3.7, 3.9, 5.1, 5.3, 6.8, 7.0, 8.2, 9.1]

    CPlotCommon.plot_line_y(y,
                               title="Y Line Plot",
                               xlabel="Index",
                               ylabel="Value",
                               saveFile=True)

    """
    CPlotCommon.plot_scatter_y(y,
                               title="Test Scatter Y Plot",
                               xlabel="Index",
                               ylabel="Value",
                               saveFile=True)
    """ 