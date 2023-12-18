import torch
import torch.nn as nn
import torch.nn.functional as F

# Suponiendo que tienes una función SSIM implementada o importada
# Por ejemplo, usando `pytorch_msssim` o una implementación personalizada
from pytorch_msssim import ssim

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        ssim_loss = 1 - ssim(y_pred, y_true, data_range=1)  # Asegúrate de ajustar data_range según tus datos
        combined_loss = self.alpha * mse + (1 - self.alpha) * ssim_loss
        return combined_loss