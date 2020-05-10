import torch
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(
        self, model, n_epochs, train_loader, test_loader, record_metrics, verbose=True
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.train_samples_size = train_loader.dataset.data.shape[0]
        self.test_loader = test_loader
        self.test_samples_size = test_loader.dataset.data.shape[0]
        self.train_sample = train_loader.dataset.data.shape[0]
        self.record_metrics = record_metrics
        self.verbose = verbose

        if record_metrics:
            self.metrics = [
                "log_p_x_given_z",
                "log_p_z_given_x",
                "log_p_x",
                "log_p_z",
                "lop_p_xz",
                "kl_prior",
                "kl_cond",
            ]

            self.losses = {"train_loss": [0] * n_epochs, "test_loss": [0] * n_epochs}

            self.train_metrics = {key: [0] * n_epochs for key in self.metrics}
            self.test_metrics = {key: [0] * n_epochs for key in self.metrics}

    @abstractmethod
    def train(self):
        pass
