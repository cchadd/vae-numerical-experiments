import torch
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, model, train_loader, test_loader, verbose=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.verbose = verbose

    @abstractmethod
    def train(self):
        pass
