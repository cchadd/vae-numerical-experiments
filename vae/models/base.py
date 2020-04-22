from abc import ABC, abstractmethod

class BaseVAE(ABC):
    
    def __init__(self):
        pass

    @abstractmethod
    def encode(self):
        pass
    
    @abstractmethod
    def decode(self):
        pass

    @abstractmethod
    def loss_function(self):
        pass