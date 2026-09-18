import torch
import torch.nn.functional as F

class CDistanceFunctions:
    # Function to compute cosine similarity between two vectors using numpy
    @staticmethod
    def cosine_distance_tensors(tensor1, tensor2):
        output = F.cosine_similarity(tensor1, tensor2, dim=0)
        output = 1 - output  # Convert similarity to distance
        return output
    
    @staticmethod
    def euclidean_distance_tensors(tensor1, tensor2):
        distance = torch.dist(tensor1, tensor2, p=2)
        return distance
    
    @staticmethod
    def cosine_similarity_vectors(vecA, vecB):
        dot_product = sum(a * b for a, b in zip(vecA, vecB))
        magnitudeA = sum(a ** 2 for a in vecA) ** 0.5
        magnitudeB = sum(b ** 2 for b in vecB) ** 0.5
        if magnitudeA == 0 or magnitudeB == 0:
            return 0.0
        return dot_product / (magnitudeA * magnitudeB)
    
    @staticmethod
    def print_distance_measures_tensors(tensor1, tensor2,text):
        cosine_sim = CDistanceFunctions.cosine_distance_tensors(tensor1, tensor2)
        euclidean_dist = CDistanceFunctions.euclidean_distance_tensors(tensor1, tensor2)
        print(f"{text} :: Similarity Cosine,Euclidean: {cosine_sim.item()},{euclidean_dist.item()}")
        