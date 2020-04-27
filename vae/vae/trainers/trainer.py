import torch
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm_notebook
from .base import BaseTrainer


class ModelTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        optimizer="adam",
        lr=1e-3,
        n_epochs=15,
        seed=1,
        record_metrics=True,
        verbose=True,
    ):
        """
            The ModelTrainer module
            
            Inputs:
            -------
            model (model):
                - The model to train. It should contain a loss_function method taken as
                convergence criteria
            train_loader (DataLoader):
                - DataLoader containng the train dataset
            test_loader (DataLoader):
                - DataLoader containing the test dataset
            optimizer (str):
                - The optimizer's name
            lr (float):
                - The learning rate to use
            n_epochs (int):
                - The number of epochs for training
            seed (int):
                - The random seed to use
            record_metrics (Bool):
                - If true the trainer will record all the metrics throughtout training and testing (log-densities, loss, kl_dvg)
            verbose (Bool):
                - Verbosity
        """
        BaseTrainer.__init__(
            self, model, n_epochs, train_loader, test_loader, record_metrics, verbose
        )

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        assert model.name in ["VAE", "HVAE"], f"{model.name} is not handled by the trainer"

        if optimizer == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr)

        else:
            raise Exception(f"Optimizer {optimizer} is not defined")

    def train(self):
        for epoch in range(self.n_epochs):
            self.__train_epoch(epoch)
            self.__test_epoch(epoch)

    def __train_epoch(self, epoch):
        self.model.train()
        train_loss = 0

        if self.verbose:
            print(f'\nTraining of epoch {epoch} in progress...')
            it = (
                enumerate(tqdm_notebook(self.train_loader))
                )

        else:
            it = enumerate(self.train_loader)

        for batch_idx, (data, _) in it:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            if self.model.name == 'VAE':
                recon_batch, z, _, mu, log_var = self.model(data)
                loss = self.model.loss_function(recon_batch, data, mu, log_var)

            elif self.model.name == 'HVAE':
                print(data.shape)
                recon_batch, z, z0, rho, gamma, mu, log_var = self.model(data)
                loss = self.model.loss_function(recon_batch, data, z0, z, rho, gamma, mu, log_var)

            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            
            if self.record_metrics:
                self.__get_model_metrics(
                    epoch,
                    recon_batch,
                    data,
                    z,
                    mu,
                    log_var,
                    sample_size=16,
                    mode="train",
                )

        if self.verbose:
            print(
                "====> Epoch: {} Average loss: {:.4f} \tLikelihood: {:.6f}".format(
                    epoch,
                    train_loss / len(self.train_loader.dataset),
                    self.train_metrics["log_p_x"][epoch],
                )
            )

    def __test_epoch(self, epoch):
        self.model.eval()
        test_loss = 0

        if self.verbose:
            print(f'\nTesting of epoch {epoch} in progress...')
            it = (
            tqdm_notebook(self.test_loader)
            )

        else:
            it = self.test_loader

        for data, _ in it:
            data = data.to(self.device)

            if self.model.name == 'VAE':
                recon, z, _, mu, log_var = self.model(data)
                # sum up batch loss
                test_loss += self.model.loss_function(recon, data, mu, log_var).item()

            elif self.model.name == 'HVAE':
                recon, z, z0, rho, gamma, mu, log_var = self.model(data)
                # sum up batch loss
                test_loss += self.model.loss_function(recon, data, z0, z, rho, gamma, mu, log_var).item()

            self.__get_model_metrics(
                epoch, recon, data, z, mu, log_var, sample_size=16, mode="test"
            )

                
        test_loss /= len(self.test_loader.dataset)

        if self.verbose:
            print("====> Test set loss: {:.4f}".format(test_loss))

    def __get_model_metrics(
        self, epoch, recon_data, data, z, mu, log_var, sample_size=16, mode="train"
    ):

        metrics = self.model.get_metrics(
            recon_data, data, z, mu, log_var, sample_size=sample_size
        )
        for key in self.metrics:
            try:

                if mode == "train":
                    self.train_metrics[key][epoch] += (
                        metrics[key].sum().item() / self.train_samples_size
                    )

                elif mode == "test":
                    self.test_metrics[key][epoch] += (
                        metrics[key].sum().item() / self.test_samples_size
                    )

            except KeyError:
                print(f"The metrics {key} is not handled")
