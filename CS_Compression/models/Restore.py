import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.dropout(out)
        out += residual
        return out

class Restore(nn.Module):
    def __init__(self):
        super(Restore, self).__init__()
        
        # Capa convolucional inicial
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Bloques residuales
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        
        # Capa convolucional final
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Aplicar relleno de replicación

        # Resto del forward
        out = self.relu1(self.conv1(x))
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.conv2(out)


        return out
