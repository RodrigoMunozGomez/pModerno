
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, output_size, CR, phi,mask):
        super(Encoder, self).__init__()
        probabilities = torch.full((input_size, output_size), CR)
        self.Phi = nn.Parameter(phi if phi is not None else torch.bernoulli(probabilities).int().float())
        self.mask = mask if mask is not None else self.Phi.data.clone().bool().to(self.Phi.device)
    def forward(self, x):
        self.Phi.data *= self.mask.to(self.Phi.device)
        x = torch.matmul(x, self.Phi.t())
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, output_size, pretrained_weights=None):
        super(Decoder, self).__init__()
        self.layer1 = nn.Linear(input_size, 2 * output_size)
        self.layer2 = nn.Linear(2 * output_size, 2 * output_size)
        self.layer3 = nn.Linear(2 * output_size, 2 * output_size)
        self.layer4 = nn.Linear(2 * output_size, output_size)

        if pretrained_weights:
            self.load_pretrained_weights(pretrained_weights)

    def load_pretrained_weights(self, pretrained_weights):
        self.layer1.weight = nn.Parameter(pretrained_weights['layer1.weight'])
        self.layer1.bias = nn.Parameter(pretrained_weights['layer1.bias'])
        self.layer2.weight = nn.Parameter(pretrained_weights['layer2.weight'])
        self.layer2.bias = nn.Parameter(pretrained_weights['layer2.bias'])
        self.layer3.weight = nn.Parameter(pretrained_weights['layer3.weight'])
        self.layer3.bias = nn.Parameter(pretrained_weights['layer3.bias'])
        self.layer4.weight = nn.Parameter(pretrained_weights['layer4.weight'])
        self.layer4.bias = nn.Parameter(pretrained_weights['layer4.bias'])

    def forward(self, x):
        
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        x = self.layer4(x)
        return x

class CSMAE(nn.Module):
    def __init__(self, output_size, CR=0.1, phi=None, pretrained_weights=None,mask = None):
        super(CSMAE, self).__init__()
        input_size = int(output_size * CR)
        self.encoder = Encoder(input_size, output_size, CR, phi,mask)
        self.decoder = Decoder(input_size * 2, output_size, pretrained_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
