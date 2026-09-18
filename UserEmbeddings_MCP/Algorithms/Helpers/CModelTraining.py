"""
Implements training for the embedding generation NN model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CModelTraining():
    @staticmethod
    def train(model, 
              x, 
              y, 
              max_epochs=50,
              min_target_loss = 1e-4, 
              lr=0.01, 
              showOutput=False):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        #-----------
        best_model = None
        current_min_loss = float('inf')
        #-----------
        model.train()
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if loss < min_target_loss:
                break
            loss.backward()
            optimizer.step()

            # Update for best model
            if loss < current_min_loss or epoch == 0:
                best_model = model
                current_min_loss = loss
                if loss < 1e-4:
                    loss = 0

            if showOutput:
                if epoch % 5 == 0:
                    print('Epoch:', epoch, 'Loss:', loss.item())
        if showOutput:
            print('Epoch:', epoch, 'Loss:', loss.item())
        return (best_model,current_min_loss) # Return end of epoch loss

   