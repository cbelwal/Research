"""
Unused legacy per-user single-layer neural-network embedding implementation.

This was previously Algorithm 2. It is retained unchanged as Algorithm 4 so
existing experimental results can still be reproduced and compared.
"""
import os
import sys

import torch
from tqdm import tqdm

topRootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(topRootPath)

from Algorithms.Helpers.CModelTraining import CModelTraining
from Algorithms.Helpers.CSingleLayer import CSingleLayer
from Algorithms.Helpers.IUserToolMatrix import IUserToolMatrix

MAX_EPOCHS = 1000
LEARNING_RATE = 0.01
SCALING_FACTOR = 1.0
MIN_TARGET_LOSS = 1e-4


def Alg_4_SingleLayerNN(
    embeddingDimensions: int = 8,
    testData: IUserToolMatrix = None,
):
    MAT_u_tau = testData.get_MAT_u_tau()
    MAT_u_tau = MAT_u_tau * SCALING_FACTOR

    print("Starting Model Training")
    MATx = torch.ones(embeddingDimensions, testData.NumberOfTools)
    MAT_E = torch.zeros(testData.NumberOfUsers, embeddingDimensions)
    loss_for_each_user = torch.zeros(testData.NumberOfUsers)

    for i in tqdm(range(testData.NumberOfUsers)):
        model = CSingleLayer(embeddingDimensions)
        tmpMAT_u_tau = MAT_u_tau[i].view(1, testData.NumberOfTools)

        model, loss = CModelTraining.train(
            model,
            MATx.T,
            tmpMAT_u_tau.T,
            max_epochs=MAX_EPOCHS,
            min_target_loss=MIN_TARGET_LOSS,
            lr=LEARNING_RATE,
        )
        loss_for_each_user[i] = loss.detach().clone()

        for name, param in model.named_parameters():
            if param.requires_grad and name == "linear.weight":
                MAT_E[i] = param.data

    return MAT_E, loss_for_each_user
