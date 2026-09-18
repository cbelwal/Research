import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.Plots.CPlotCommon import CPlotCommon
import numpy as np

class CPolynomialFitReduction:
    def __init__(self,embeddingDimensions:int=8):
        self.embeddingDimensions = embeddingDimensions
        
    def get_reduced_dimension_polynomial_fit(self, data: np.ndarray) -> np.ndarray:
        no_of_tools = data.shape[1]
        x = np.linspace(0, no_of_tools - 1, no_of_tools)
        y = data.flatten()

        fitted_ploy,residuals,_,_,_ = np.polyfit(x, y, deg=self.embeddingDimensions - 1,full=True)
        model = np.poly1d(fitted_ploy)

        # NOTE: In case of a perfect fit, the residuals will be an empty array
        # So return 0 in that case.
        if len(residuals) == 0: 
            residuals = np.zeros(1)        

        return (model.coefficients, residuals)
       


    """
    @staticmethod
    def reduce_dimension_polynomial_fit(data: np.ndarray, degree: int, num_points: int) -> np.ndarray:
        '''
        Reduces the dimensionality of the input data using polynomial fitting.
        
        Parameters:
        - data: np.ndarray, shape (n_samples, n_features)
            The input data to be reduced.
        - degree: int
            The degree of the polynomial to fit.
        - num_points: int
            The number of points to sample from the fitted polynomial.
        
        Returns:
        - reduced_data: np.ndarray, shape (n_samples, num_points)
            The dimensionally reduced data.
        '''
        n_samples, n_features = data.shape
        reduced_data = np.zeros((n_samples, num_points))
        
        x_new = np.linspace(0, n_features - 1, num_points)
        
        for i in range(n_samples):
            coeffs = np.polyfit(np.arange(n_features), data[i, :], degree)
            poly = np.poly1d(coeffs)
            reduced_data[i, :] = poly(x_new)
        
        return reduced_data
    """