import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, grad
from torchvision.utils import save_image
from .base import BaseVAE


class VAE(BaseVAE, nn.Module):
    def __init__(self, model_type="mlp", latent_dim=2):

        BaseVAE.__init__(self)
        nn.Module.__init__(self)
        if model_type == "mlp":
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
        self.latent_dim = latent_dim
        self.normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(latent_dim).to(self.device),
            covariance_matrix=torch.eye(latent_dim).to(self.device),
        )

    def forward(self, x):

        mu, log_var = self.encode(x.view(-1, 784))
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decode(z)
        return recon_x, z, eps, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    def encode(self, x):
        return self.__encoder(x)

    def decode(self, z):
        x_prob = self.__decoder(z)
        return x_prob

    def sample_img(self, z):
        """
        Simulate p(x|z) to generate an image
        """
        x_prob = self.decode(z)
        return torch.distributions.Bernoulli(probs=x_prob).sample()

    def __encode_mlp(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def __decode_mlp(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def _sample_gauss(self, mu, std):
        # Reparametrization trick

        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    ########## Estimate densities ##########

    def get_metrics(self, recon_x, x, z, mu, log_var, sample_size=16):
        """
        Estimates all metrics '(log-densities, loss, kl-dvg)

        Output:
        -------
        
        log_densities (dict): Dict with keys [
            'log_p_x_given_z',
            'log_p_z_given_x',
            'log_p_x'
            'log_p_z'
            'lop_p_xz'
            ]
        """
        metrics = {}

        metrics["log_p_x_given_z"] = self.log_p_x_given_z(recon_x, x)
        metrics["log_p_z_given_x"] = self.log_p_z_given_x(
            z, recon_x, x, sample_size=sample_size
        )
        metrics["log_p_x"] = self.log_p_x(x, sample_size=sample_size)
        metrics["log_p_z"] = self.log_z(z)
        metrics["lop_p_xz"] = self.log_p_xz(recon_x, x, z)
        metrics["kl_prior"] = self.kl_prior(mu, log_var)
        metrics["kl_cond"] = self.kl_cond(
            recon_x, x, z, mu, log_var, sample_size=sample_size
        )
        return metrics

    def log_p_x_given_z(self, recon_x, x, reduction="none"):
        """
        Estimate the decoder's log-density modelled as follows:
            p(x|z)     = \prod_i Bernouilli(x_i|pi_{theta}(z_i))
            p(x = s|z) = \prod_i (pi(z_i))^x_i * (1 - pi(z_i)^(1 - x_i))"""
        return -F.binary_cross_entropy(
            recon_x, x.view(-1, 784), reduction=reduction
        ).sum(dim=1)

    def log_z(self, z):
        """
        Return Normal density function as prior on z
        """
        return self.normal.log_prob(z)

    def log_p_x(self, x, sample_size=16):
        """
        Estimate log(p(x)) using importance sampling with q(z|y)
        """

        mu, log_var = self.encode(x.view(-1, 784))
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=self.device)
        Z = (mu + Eps * torch.exp(0.5 * log_var)).reshape(-1, self.latent_dim)
        recon_X = self.decode(Z)
        bce = F.binary_cross_entropy(
            recon_X, x.view(-1, 784).repeat(sample_size, 1), reduction="none"
        )

        # compute densities to recover p(x)
        logpxz = -bce.reshape(sample_size, -1, 784).sum(dim=2)  # log(p(x|z))
        logpz = self.log_z(Z).reshape(sample_size, -1)  # log(p(z))
        logqzx = (
            torch.distributions.MultivariateNormal(
                loc=mu, covariance_matrix=torch.diag_embed(torch.exp(log_var))
            )
            .log_prob(Z.reshape(sample_size, -1, self.latent_dim))
            .reshape(sample_size, -1)
        )  # log(q(z|x))
        logpx = (logpxz + logpz - logqzx).logsumexp(dim=0) - torch.log(
            torch.Tensor([sample_size]).to(self.device)
        )
        return logpx

    def log_p_z_given_x(self, z, recon_x, x, sample_size=16):
        """
        Estimate log(p(z|x)) using Bayes rule and Importance Sampling for log(p(x))
        """
        logpx = self.log_p_x(x, sample_size)
        lopgxz = self.log_p_x_given_z(recon_x, x)
        logpz = self.log_z(z)
        return lopgxz + logpz - logpx

    def log_p_xz(self, recon_x, x, z):
        """
        Estimate log(p(x, z)) using Bayes rule
        """
        logpxz = self.log_p_x_given_z(recon_x, x)
        logpz = self.log_z(z)
        return logpxz + logpz

    ########## Kullback-Leiber divergences estimates ##########

    def kl_prior(self, mu, log_var):
        """KL[q(z|y) || p(z)] : exact formula"""
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def kl_cond(self, recon_x, x, z, mu, log_var, sample_size=16):
        """
        KL[p(z|x) || q(z|x)]

        Note:
        -----
        p(z|x) is approximated using IS on log(p(x))
        """
        logpzx = self.log_p_z_given_x(z, recon_x, x, sample_size=sample_size)
        logqzx = logqzx = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=torch.diag_embed(torch.exp(log_var))
        ).log_prob(z)

        return (logqzx - logpzx).sum()

    # def log_p_x(self, x):
    #    mu, log_var = self.encode(x.view(-1, 784))
    #    eps = torch.randn()


#
# def log_p_zx(self, z):


class HVAE(VAE):
    def __init__(
        self,
        n_lf=3,
        eps_lf=0.01,
        beta_zero=1,
        tempering="fixed",
        model_type="mlp",
        latent_dim=2,
    ):
        """
        Inputs:
        -------

        n_lf (int): Number of leapfrog steps to perform
        eps_lf (float): Leapfrog step size
        beta_zero (float): Initial tempering
        tempering (str): Tempering type (free, fixed)
        model_type (str): Model type for VAR (mlp, convnet)
        latent_dim (int): Latentn dimension
        """
        VAE.__init__(self, model_type=model_type, latent_dim=latent_dim)

        self.vae_forward = super().forward

        self.n_lf = n_lf
        self.eps_lf = torch.Tensor([eps_lf]).to(self.device)

        assert 0 < beta_zero < 1, "Tempering factor should belong to [0, 1]"

        if tempering == "fixed":

            self.beta_zero_sqrt = nn.Parameter(torch.Tensor([beta_zero]))
            self.tempering = tempering

        elif tempering == "free":

            raise NotImplementedError()

        def forward(self, x):
            """
            Perform Hamiltonian Importance Sampling
            """

            recon_x, z0, eps, _, log_var = self.vae_forward(x)
            gamma = torch.randn_like(z0, device=self.device)
            rho = gamma / self.beta_zero_sqrt
            z = z0
            beta_sqrt_old = self.beta_zero_sqrt

            for k in range(self.n_lf):

                # Perform leapforog steps

                recon_x = self.decode(z)

                # Computes potential energy
                U = -self.log_p_xz(recon_x, x, z).sum()

                # Compute its gradient
                g = grad(U, z)[0]

                # 1st leapfrog step
                rho_ = rho - (self.eps_lf / 2) * g
                z = z + self.eps_lf * rho_

                # 2nd leapfrog ste
                recon_x = self.decode(z)
                U = -self.log_p_xz(recon_x, x, z).sum()
                g = grad(U, z)[0]

                rho__ = rho_ - (self.eps_lf / 2) * g

                # tempering steps
                beta_sqrt = self.__tempering(k)
                rho = (beta_sqrt / beta_sqrt_old) * rho__
                beta_sqrt_old = beta_sqrt

            return recon_x, z, z0, rho, eps, gamma, log_var

        # def __log_p(self, x, )

        def __tempering(self, k):
            """Perform tempering step"""

            beta_k = (
                1 - (1 / self.beta_zero_sqrt) * k / self.n_lf ** 2
            ) + 1 / self.beta_zero_sqrt

            return 1 / beta_k
