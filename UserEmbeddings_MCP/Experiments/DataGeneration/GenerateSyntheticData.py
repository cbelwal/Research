import os,sys
# ----------------------------------------------
# Ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------

from Experiments.DataGeneration.CGenerateSyntheticData import CGenerateSyntheticData
from Experiments.Database.CDatabaseManager import CDatabaseManager   

if __name__ == "__main__":
    CDatabaseManager.delete_db_file()
    dataGenerator = CGenerateSyntheticData()
    dataGenerator.generate_synthetic_data()
