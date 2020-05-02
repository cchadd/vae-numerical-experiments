import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, grad
from torchvision.utils import save_image
from .base import BaseVAE


# TO BE cleaned 
class VAE(BaseVAE, nn.Module):

    def __init__(self, model_type='mlp', input_dim=2, latent_dim=2):

        BaseVAE.__init__(self)
        nn.Module.__init__(self)

        self.name = "VAE"
        self.archi = "Gauss" 

        if model_type == "mlp":

            # Encoder network
            self.fc1 = nn.Linear(input_dim, 64)

            # \mu_{phi}
            self.fc21 = nn.Linear(64, 32)
            self.fc31 = nn.Linear(32, latent_dim)

            # \sigma_{\phi}
            self.fc22 = nn.Linear(64, 32)
            self.fc32 = nn.Linear(32, latent_dim)

            # \mu_{\theta}
            self.fc41 = nn.Linear(latent_dim, 32)
            self.fc51 = nn.Linear(32, 64)
            self.fc61 = nn.Linear(64, input_dim)

            # \sigma_{\theta}
            self.fc42 = nn.Linear(latent_dim, 32)
            self.fc52 = nn.Linear(32, 64)
            self.fc62 = nn.Linear(64, input_dim)
 
            self.__encoder = self.__encode_mlp
            self.__decoder = self.__decode_mlp


        else:
            raise Exception(f"Architecture {model_type} is not defined")

        self.model_type = model_type
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(latent_dim).to(self.device),
            covariance_matrix=torch.eye(latent_dim).to(self.device),
        ) 


    def forward(self, x, ensure_geo=False):
        """
        ensure_geo (Bool): - If True, the model will only update \mu_{\phi}, \sigma_{\phi}, \mu_{\ theta}
                           - If False, the model will only update \sigma_{\ theta} like 'Arvanitidis, 2018'
        """
        mu, log_var = self.encode(x.view(-1, self.input_dim))

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_mu, recon_log_var = self.decode(z)

        #if ensure_geo:
        #    recon_log_var = recon_log_var.detach()

        #else:
        #    mu = mu.detach()
        #    log_var = log_var.detach()
        #    recon_mu = recon_mu.detach()

        return (recon_mu, recon_log_var), z, eps, mu, log_var


    def loss_function(self, x, recon_mu, recon_log_var, mu, log_var):
        #BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_dim), reduction="sum")
        LIK = -torch.distributions.MultivariateNormal(
            loc=recon_mu,
            covariance_matrix=torch.diag_embed(recon_log_var.exp())
        ).log_prob(x.view(-1, self.input_dim)).sum()
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return LIK + 0.3 * KLD

    def encode(self, x):
        return self.__encoder(x)

    def decode(self, z):
        return self.__decoder(z)

    def sample_img(self, z):
        """
        Simulate p(x|z) to generate an image
        """
        recon_mu, recon_log_var = self.decode(z)
        return torch.distributions.MultivariateNormal(loc=recon_mu, covariance_matrix=torch.diag_embed(recon_log_var.exp())).sample()

    def __encode_mlp(self, x):
        h1 = torch.tanh(self.fc1(x))
        h21 = torch.tanh(self.fc21(h1))
        h22 = torch.tanh(self.fc22(h1))
        h31 = torch.tanh(self.fc31(h21))
        h32 = torch.tanh(self.fc32(h22))

        return F.softplus(h31), F.softplus(h32)

    def __decode_mlp(self, z):
        h41 = torch.tanh(self.fc41(z))
        h42 = torch.tanh(self.fc42(z))
        h51 = torch.tanh(self.fc51(h41))
        h52 = torch.tanh(self.fc52(h42))
        h61 = torch.tanh(self.fc61(h51))
        h62 = torch.tanh(self.fc62(h52))
        
        return (torch.sigmoid(h61), F.softplus(h62))


    def _sample_gauss(self, mu, std):
        # Reparametrization trick

        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    ########## Estimate densities ##########

    def get_metrics(self, recon_mu, recon_log_var, x, z, mu, log_var, sample_size=16):
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

        metrics["log_p_x_given_z"] = self.log_p_x_given_z(x, recon_mu, recon_log_var)
        metrics["log_p_z_given_x"] = self.log_p_z_given_x(
            z, x, recon_mu, recon_log_var, sample_size=sample_size
        )
        metrics["log_p_x"] = self.log_p_x(x, sample_size=sample_size)
        metrics["log_p_z"] = self.log_z(z)
        metrics["lop_p_xz"] = self.log_p_xz(x, z, recon_mu, recon_log_var)
        metrics["kl_prior"] = self.kl_prior(mu, log_var)
        metrics["kl_cond"] = self.kl_cond(
            x, z, recon_mu, recon_log_var, mu, log_var, sample_size=sample_size
        )
        return metrics

    def log_p_x_given_z(self, x, recon_mu, recon_log_var):
        """
        Estimate the decoder's log-density modelled as follows:
            p(x|z)     = \prod_i Normal(x_i|\mu_{theta}(z_i), \sigma_{theta}(z_i))
        """
        return torch.distributions.MultivariateNormal(
            loc=recon_mu,
            covariance_matrix=torch.diag_embed(recon_log_var.exp())
        ).log_prob(x.view(-1, self.input_dim)).sum()


    def log_z(self, z):
        """
        Return Normal density function as prior on z
        """
        return self.normal.log_prob(z)

    def log_p_x(self, x, sample_size=16):
        """
        Estimate log(p(x)) using importance sampling with q(z|x)
        """

        mu, log_var = self.encode(x.view(-1, self.input_dim))
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=self.device)
        Z = (mu + Eps * torch.exp(0.5 * log_var)).reshape(-1, self.latent_dim)
        
        recon_mu, recon_log_var = self.decode(Z)

        lik = torch.distributions.MultivariateNormal(
            loc=recon_mu,
            covariance_matrix=torch.diag_embed(recon_log_var.exp())
        ).log_prob(x.view(-1, self.input_dim).repeat(sample_size, 1))


        # compute densities to recover p(x)
        logpxz = lik.reshape(sample_size, -1)  # log(p(x|z))
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

    def log_p_z_given_x(self, z, x, recon_mu, recon_log_var, sample_size=16):
        """
        Estimate log(p(z|x)) using Bayes rule and Importance Sampling for log(p(x))
        """
        logpx = self.log_p_x(x, sample_size)
        logpxz = self.log_p_x_given_z(x, recon_mu, recon_log_var)
        logpz = self.log_z(z)
        return logpxz + logpz - logpx

    def log_p_xz(self, x, z, recon_mu, recon_log_var):
        """
        Estimate log(p(x, z)) using Bayes rule
        """
        logpxz = self.log_p_x_given_z(x, recon_mu, recon_log_var)
        logpz = self.log_z(z)
        return logpxz + logpz

    ########## Kullback-Leiber divergences estimates ##########

    def kl_prior(self, mu, log_var):
        """KL[q(z|x) || p(z)] : exact formula"""
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def kl_cond(self, x, z, recon_mu, recon_log_var, mu, log_var, sample_size=16):
        """
        KL[p(z|x) || q(z|x)]

        Note:
        -----
        p(z|x) is approximated using IS on log(p(x))
        """
        logpzx = self.log_p_z_given_x(z, x, recon_mu, recon_log_var, sample_size=sample_size)
        logqzx = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=torch.diag_embed(torch.exp(log_var))
        ).log_prob(z)

        return (logqzx - logpzx).sum()