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
        seed=1
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
            self, model, train_loader, test_loader
        )

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        assert model.name in [
            "VAE",
            "HVAE",
            "RHVAE",
            "adaRHVAE"
        ], f"{model.name} is not handled by the trainer"

        if optimizer == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr)

        else:
            raise Exception(f"Optimizer {optimizer} is not defined")

    def train(self, n_epochs, only_train=False, record_metrics=False, verbose=False):

        self.only_train = only_train
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

            self.train_metrics = {key: [0] * n_epochs for key in self.metrics}
            self.test_metrics = {key: [0] * n_epochs for key in self.metrics}

        for epoch in range(n_epochs):

            self.__train_epoch(epoch)
            if not self.only_train:
                self.__test_epoch(epoch)

            self.total_epoch += 1

            if epoch % 50 == 0:
                print(f'Epoch {epoch} \tLoss: { self.losses["train_loss"][-1]}')

    def __train_epoch(self, epoch):
        self.model.train()
        train_loss = 0

        if self.verbose:
            print(f"\nTraining of epoch {epoch} in progress...")
            it = enumerate(tqdm_notebook(self.train_loader))

        else:
            it = enumerate(self.train_loader)

        for batch_idx, (data, _) in it:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            # Train Bernoulli models
            if self.model.archi == "Bernoulli":

                if self.model.name == "VAE":
                    recon_batch, z, _, mu, log_var = self.model(data)
                    loss = self.model.loss_function(recon_batch, data, mu, log_var)

                elif self.model.name == "HVAE":
                    recon_batch, z, z0, rho, eps0, gamma, mu, log_var = self.model(data)
                    loss = self.model.loss_function(
                        recon_batch, data, z0, z, rho, eps0, gamma, mu, log_var
                    )

                elif self.model.name == "RHVAE":
                    recon_batch, z, z0, rho, eps0, gamma, mu, log_var, G = self.model(data)
                    loss = self.model.loss_function(
                        recon_batch, data, z0, z, rho, eps0, gamma, mu, log_var, G
                    )

                elif self.model.name == "adaRHVAE":
                    recon_batch, z, z0, rho, eps0, gamma, mu, log_var, G, G_log_det = self.model(data)

                    loss = self.model.loss_function(
                        recon_batch, data, z0, z, rho, eps0, gamma, mu, log_var, G, G_log_det
                    )
                    #if loss > 1000000:
                    #    print('Loss exceeds 1000000 :')
                    #    print('G', G)
                    #    print('LogDetG', G)
                    #    print('Logvar', log_var)

                elif self.model.name == 'Two times':
                    if epoch < int(self.n_epochs / 2):
                        recon_batch, z, z0, rho, gamma, mu, log_var = self.model(data, encoder_only=True)
                        loss = self.model.loss_function(
                            recon_batch, data, z0, z, rho, gamma, mu, log_var
                        )
                    else:
                        recon_batch, z, z0, rho, gamma, mu, log_var = self.model(data, decoder_only=True)
                        loss = self.model.loss_function(
                            recon_batch, data, z0, z, rho, gamma, mu, log_var
                        )



            elif self.model.archi == "Gauss":

                if self.model.name == "VAE":
                    # if self.n_epochs < self.n_epochs / 2:
                    recon_batch, z, _, mu, log_var = self.model(data)

                    # else:
                    #    recon_batch, z, _, mu, log_var = self.model(data, ensure_geo=False)

                    loss = self.model.loss_function(recon_batch, data, mu, log_var)


            train_loss += loss

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

        # Average loss over batches
        train_loss /= len(self.train_loader.batch_sampler)

        train_loss.backward()

        # Grad clipping
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
                
        self.optimizer.step()

        if self.model.name == "adaRHVAE" and self.model.metric == 'TBL':
                self.model.update_metric()

        self.losses["train_loss"].append(train_loss.item())

        if self.verbose:
            if self.record_metrics:
                print(
                    "====> Epoch: {} Average loss: {:.4f} \tLikelihood: {:.6f} \t KL prior: {:.4f}".format(
                        epoch,
                        train_loss,
                        self.train_metrics["log_p_x"][epoch],
                        self.train_metrics["kl_prior"][epoch],
                    )
                )

            else:
                print(
                    "====> Epoch: {} Average loss: {:.4f}".format(
                        epoch,
                        train_loss
                    )
                )


    def __test_epoch(self, epoch):
        self.model.eval()
        test_loss = 0

        if self.verbose:
            print(f"\nTesting of epoch {epoch} in progress...")
            it = tqdm_notebook(self.test_loader)

        else:
            it = self.test_loader

        for data, _ in it:
            data = data.to(self.device)

            if self.model.archi == "Bernoulli":

                if self.model.name == "VAE":
                    recon, z, _, mu, log_var = self.model(data)

                    loss = self.model.loss_function(
                        recon, data, mu, log_var
                    ).item()

                elif self.model.name == "HVAE":
                    recon, z, z0, rho, eps0, gamma, mu, log_var = self.model(data)

                    loss = self.model.loss_function(
                        recon, data, z0, z, rho, eps0, gamma, mu, log_var
                    ).item()

                elif self.model.name == "RHVAE":
                    recon, z, z0, rho, eps0, gamma, mu, log_var, G = self.model(data)

                    loss = self.model.loss_function(
                        recon, data, z0, z, rho, eps0, gamma, mu, log_var, G
                    ).item()

                elif self.model.name == "adaRHVAE":
                    recon, z, z0, rho, eps0, gamma, mu, log_var, G, G_log_det = self.model(data)

                    loss = self.model.loss_function(
                        recon, data, z0, z, rho, eps0, gamma, mu, log_var, G, G_log_det
                    ).item()

            elif self.model.archi == "Gauss":

                if self.model.name == "VAE":

                    recon, z, _, mu, log_var = self.model(data)

                    loss = self.model.loss_function(
                        recon, data, mu, log_var
                    ).item()

            test_loss += loss

            if self.record_metrics:
                self.__get_model_metrics(
                    epoch, recon, data, z, mu, log_var, sample_size=16, mode="test"
                )

        test_loss /= len(self.test_loader.batch_sampler)

        self.losses["test_loss"].append(test_loss)

        if self.verbose:
            if self.record_metrics:
                print(
                    "====> Test set loss: {:.4f} \tLikelihood: {:.4f}".format(
                        test_loss, self.test_metrics["log_p_x"][epoch]
                    )
                )
            
            else:
                print(
                    "====> Test set loss: {:.4f}".format(
                        test_loss
                    )
                )


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
                        metrics[key].sum().item() / len(self.train_loader.batch_sampler)
                    )

                elif mode == "test":
                    self.test_metrics[key][epoch] += (
                        metrics[key].sum().item() / len(self.test_loader.batch_sampler)
                    )

            except KeyError:
                print(f"The metrics {key} is not handled")
