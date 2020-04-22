import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import random
from base import BASE_VAE


class VAE(BASE_VAE, nn.Module):

    def __init__(self,
                 model_type='mlp',
                 latent_dim=2,
                 device='cpu'):
        
        BASE_VAE.__init__(self)
        nn.Module.__init__(self)
        
        if model_type == 'mlp':
            # encoder network
            self.fc1 = nn.Linear(784, 400)
            self.fc21 = nn.Linear(400, latent_dim)
            self.fc22 = nn.Linear(400, latent_dim)
            
            # decoder network
            self.fc3 = nn.Linear(latent_dim, 400)
            self.fc4 = nn.Linear(400, 784)
            
            self.__encoder = self.__encode_mlp
            self.__decoder = self.__decode_mlp

        else:
            raise Exception(f"Architecture {model_type} is not defined")

        self.model_type = model_type
        self.device = device

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        std = torch.exp(0.5 * log_var)
        z = self._sample_gauss(mu, std)
        
        return self.decode(z), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    def encode(self, x):
        return self.__encoder(x)
    
    def decode(self, z):
        return self.__decoder(z)
    
    def __encode_mlp(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def __decode_mlp(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        
        # Sample N(O, I)
        eps = torch.randn_like(std)
        return mu + eps * std