import torch
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(
        self, model, train_loader, test_loader, verbose=True
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.train_samples_size = train_loader.dataset.data.shape[0]
        self.test_loader = test_loader
        self.test_samples_size = test_loader.dataset.data.shape[0]
        self.train_sample = train_loader.dataset.data.shape[0]
        self.total_epoch = 0

        self.losses = {"train_loss": [], "test_loss": []}

    @abstractmethod
    def train(self):
        pass
