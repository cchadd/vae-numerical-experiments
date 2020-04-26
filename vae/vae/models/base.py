from abc import ABC, abstractmethod
import torch


class BaseVAE(ABC):
    def __init__(self):

         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass

    @abstractmethod
    def loss_function(self):
        pass
