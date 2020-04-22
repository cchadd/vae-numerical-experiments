import torch
import torch.optim as optim
import numpy as np
import random
from .base import BaseTrainer


class ModelTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        optimizer="adam",
        lr=1e-3,
        batch_size=100,
        n_epochs=15,
        seed=1,
        verbose=True,
    ):
        """
            The ModelTrainer module
            
            Inputs:
            -------
            model (model):
                - The model to train. It should contain a loss_fonction method taken as
                convergence criteria
            train_loader (DataLoader):
                - DataLoader containng the train dataset
            test_loader (DataLoader):
                - DataLoader containing the test dataset
            optimizer (str):
                - The optimizer's name
            lr (float):
                - The learning rate to use
            batch_size (int):
                - The batch_size for training
            n_epochs (int):
                - The number of epochs for training
            seed (int):
                - The random seed to use
            verbose (Bool):
                - Verbosity
        """

        BaseTrainer.__init__(self, model, train_loader, test_loader, verbose)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.verbose = True

        if optimizer == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr)

        else:
            raise Exception(f"Optimizer {optimizer} is not defined")

    def train(self):
        for epoch in range(self.n_epochs):
            self.__train_epoch(epoch)
            self.__test_epoch()

    def __train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)
            loss = self.model.loss_function(recon_batch, data, mu, log_var)

            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            if batch_idx % 100 == 0 and self.verbose:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        loss.item() / len(data),
                    )
                )

        if self.verbose:
            print(
                "====> Epoch: {} Average loss: {:.4f}".format(
                    epoch, train_loss / len(self.train_loader.dataset)
                )
            )

    def __test_epoch(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                recon, mu, log_var = self.model(data)

                # sum up batch loss
                test_loss += self.model.loss_function(recon, data, mu, log_var).item()

        test_loss /= len(self.test_loader.dataset)

        if self.verbose:
            print("====> Test set loss: {:.4f}".format(test_loss))
