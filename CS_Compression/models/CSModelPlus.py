import torch    
import torch.nn as nn
import torch.nn.functional as F


class CSModelPlus(nn.Module):
    def __init__(self,output_size,CR=0.1, phi=None,pretrained_weights=None,device="cuda:0"):
        super(CSModelPlus, self).__init__()
        self.device = device
        input_size = int(output_size*CR)
        probabilities = torch.full((input_size,output_size),CR)
        self.Phi = phi if phi is not None else torch.bernoulli(probabilities).int().float().to(self.device)

        self.layer1 = nn.Linear(input_size, 2 * output_size)
        self.layer2 = nn.Linear(2 * output_size, 2 * output_size)
        self.layer3 = nn.Linear(2 * output_size, 2 * output_size)  
        self.layer4 = nn.Linear(2 * output_size, output_size)  

        if pretrained_weights:
            self.layer1.weight = nn.Parameter(pretrained_weights['layer1.weight'])
            self.layer1.bias = nn.Parameter(pretrained_weights['layer1.bias'])
            self.layer2.weight = nn.Parameter(pretrained_weights['layer2.weight'])
            self.layer2.bias = nn.Parameter(pretrained_weights['layer2.bias'])
            self.layer3.weight = nn.Parameter(pretrained_weights['layer3.weight'])
            self.layer3.bias = nn.Parameter(pretrained_weights['layer3.bias'])
            self.layer4.weight = nn.Parameter(pretrained_weights['layer4.weight'])
            self.layer4.bias = nn.Parameter(pretrained_weights['layer4.bias'])

    def forward(self, x):

        x = torch.matmul(x, self.Phi.t().to(self.device))
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        x = self.layer4(x)
        return x
