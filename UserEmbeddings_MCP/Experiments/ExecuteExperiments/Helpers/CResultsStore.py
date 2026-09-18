"""
Data is stored as torch tensors
"""
import os,sys
import torch
import pickle
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(topRootPath)
#----------------------------------------------

from Experiments.CConfig import CConfig

class CResultsStore:
    def __init__(self,algID:int):
        self.algID = algID
    
    def get_folder_path(self) -> str:
        folderPath =    os.path.dirname(
                        os.path.dirname( #Experiments
                        os.path.dirname( #UserEmbeddings
                        os.path.abspath(__file__))))
        dbFolder = os.path.join(folderPath, "Data")
        return dbFolder

    # Insert the algorithm ID into the file name
    def get_file_name_for_algorithm(self, fileName) -> str:
        fileNameComponents = fileName.split(".")
        fileName = f"{fileNameComponents[0]}_alg_{self.algID}.{fileNameComponents[1]}"
        return fileName

    def get_file_path(self,baseFileName) -> str:
        fileName = self.get_file_name_for_algorithm(baseFileName)
        folderPath = self.get_folder_path()
        dbFolder = os.path.join(folderPath, "ExperimentResults")
        dbFilePath = os.path.join(dbFolder, fileName)
        return dbFilePath

    def store_embeddings(self, MAT_E):
        filePath = self.get_file_path(CConfig.BASE_EMBEDDINGS_FILE_NAME)
        torch.save(MAT_E, filePath)
    
    def load_embeddings(self):
        filePath = self.get_file_path(CConfig.BASE_EMBEDDINGS_FILE_NAME)
        MAT_E = torch.load(filePath)
        return MAT_E
    
    def store_training_loss(self, loss_for_each_user):
        fileName = self.get_file_path(CConfig.BASE_TRAINING_LOSS_FILE_NAME)
        loss_file_path = os.path.join(self.get_folder_path(), fileName)
        with open(loss_file_path, 'wb') as f:
            pickle.dump(loss_for_each_user, f)
        print(f"Training loss stored in file: {loss_file_path}")

    def load_training_loss(self):
        loss_file_path = self.get_file_path(CConfig.BASE_TRAINING_LOSS_FILE_NAME)
        with open(loss_file_path, 'rb') as f:
            loss_for_each_user = pickle.load(f)
        print(f"Training loss loaded from file: {loss_file_path}")
        return loss_for_each_user

if __name__ == "__main__":
    algID = 3
    # ---------- Test storing and loading embeddings
    MAT_E_Orig = torch.randn(10, 8)  # Example: 10 users, 8-dimensional embeddings
    store = CResultsStore()
    store.store_embeddings(MAT_E_Orig, algID=algID)
    print(f"Embeddings stored in file: {MAT_E_Orig[0]}")
    
    # Load embeddings back
    MAT_E = store.load_embeddings(algID=algID)
    print(f"Loaded Embeddings: {MAT_E[0]}")

    # Validation
    for i in range(MAT_E_Orig.shape[0]):
        assert torch.allclose(MAT_E_Orig[i], MAT_E[i]), "Embeddings do not match!"
    print("Embeddings matched successfully!")
    
    # ---------- Test storing and loading training loss
    loss_for_each_user_orig = [0.1 * i for i in range(10)]  # Example losses for 10 users
    store.store_training_loss(loss_for_each_user_orig, algID=algID)
    print(f"Training loss stored in file: {loss_for_each_user_orig}")

    # ---------- Load training loss back
    loss_for_each_user = store.load_training_loss(loss_for_each_user_orig, algID=algID)
    print(f"Loaded Training Loss: {loss_for_each_user}")  
    # Validation
    assert loss_for_each_user_orig == loss_for_each_user, "Training losses do not match!"
    