import torch
from abc import ABC, abstractmethod

class BaseMCMCSampler(ABC):

    def __init__(self, chain_size, dim, seed=None):

        self.chain_size = chain_size
        self.dim = dim
        torch.manual_seed(seed)


    @abstractmethod
    def sample(self):
        pass