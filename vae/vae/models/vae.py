import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.functional import jacobian as jac
from torchvision import datasets, transforms
from torch.autograd import Variable, grad
from torchvision.utils import save_image
from .base import BaseVAE
import copy


class VAE(BaseVAE, nn.Module):
    def __init__(self, model_type="mlp", input_dim=784, latent_dim=2):

        BaseVAE.__init__(self)
        nn.Module.__init__(self)

        self.name = "VAE"
        self.archi = "Bernoulli"

        if model_type == "convnet":
            self.conv1 = nn.Conv2d(1, 16, 5, 2, padding=2)
            self.conv2 = nn.Conv2d(16, 32, 5, 2, padding=2)
            self.conv3 = nn.Conv2d(32, 32, 5, 2, padding=2)
            self.fc1 = nn.Linear(512, 450)
            self.fc2 = nn.Linear(450, latent_dim)
            self.fc22 = nn.Linear(450, latent_dim)

            self.fc3 = nn.Linear(latent_dim, 450)
            self.fc4 = nn.Linear(450, 512)
            self.upsample = nn.Upsample(scale_factor=2)
            self.deconv1 = nn.ConvTranspose2d(32, 32, 5, 2, padding=2)
            self.deconv2 = nn.ConvTranspose2d(32, 16, 5, 2, padding=2, output_padding=1)
            self.deconv3 = nn.ConvTranspose2d(16, 1, 5, 2, padding=2, output_padding=1)

            self.__encoder = self.__encode_convnet
            self.__decoder = self.__decode_convnet

        elif model_type == "mlp":
            # encoder network
            self.fc1 = nn.Linear(input_dim, 400)
            self.fc21 = nn.Linear(400, latent_dim)
            self.fc22 = nn.Linear(400, latent_dim)

            # decoder network
            self.fc3 = nn.Linear(latent_dim, 400)
            self.fc4 = nn.Linear(400, input_dim)

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

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.input_dim))
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decode(z)
        return recon_x, z, eps, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(
            recon_x, x.view(-1, self.input_dim), reduction="sum"
        )
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return BCE + KLD

    def encode(self, x):
        return self.__encoder(x)

    def decode(self, z):
        x_prob = self.__decoder(z)
        return x_prob

    def sample_img(self, z=None, n_samples=1):
        """
        Simulate p(x|z) to generate an image
        """

        if z is None:
            z = self.normal.sample(sample_shape=(n_samples,)).to(self.device)

        else:
            n_samples = z.shape[0]

        z.requires_grad_(True)

        x_prob = self.decode(z)
        return x_prob  # torch.distributions.Bernoulli(probs=x_prob).sample()

    def __encode_convnet(self, x):
        x = F.relu(self.conv1(x.view(-1, 1, 28, 28)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        return self.fc2(x), self.fc22(x)

    def __encode_mlp(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def __decode_convnet(self, z):
        x = F.relu(self.fc3(z))
        x = F.relu(self.fc4(x)).view(-1, 32, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x.view(-1, self.input_dim)

    def __decode_mlp(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def _sample_gauss(self, mu, std):
        # Reparametrization trick

        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    ########## Estimate densities ##########

    def get_metrics(self, recon_x, x, z, mu, log_var, sample_size=10):
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
            recon_x, x.view(-1, self.input_dim), reduction=reduction
        ).sum(dim=1)

    def log_z(self, z):
        """
        Return Normal density function as prior on z
        """
        return self.normal.log_prob(z)

    def log_p_x(self, x, sample_size=10):
        """
        Estimate log(p(x)) using importance sampling with q(z|x)
        """

        mu, log_var = self.encode(x.view(-1, self.input_dim))
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=self.device)
        Z = (mu + Eps * torch.exp(0.5 * log_var)).reshape(-1, self.latent_dim)
        recon_X = self.decode(Z)
        bce = F.binary_cross_entropy(
            recon_X, x.view(-1, self.input_dim).repeat(sample_size, 1), reduction="none"
        )

        # compute densities to recover p(x)
        logpxz = -bce.reshape(sample_size, -1, self.input_dim).sum(dim=2)  # log(p(x|z))
        logpz = self.log_z(Z).reshape(sample_size, -1)  # log(p(z))
        logqzx = (
            torch.distributions.MultivariateNormal(
                loc=mu, covariance_matrix=torch.diag_embed(torch.exp(log_var))
            )
            .log_prob(Z.reshape(sample_size, -1, self.latent_dim))
            .reshape(sample_size, -1)
        )  # log(q(z|x))
        logpx = (logpxz + logpz - logqzx).logsumexp(dim=0).mean(dim=0) - torch.log(
            torch.Tensor([sample_size]).to(self.device)
        )
        return logpx

    def log_p_z_given_x(self, z, recon_x, x, sample_size=10):
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

    def kl_cond(self, recon_x, x, z, mu, log_var, sample_size=10):
        """
        KL[p(z|x) || q(z|x)]

        Note:
        -----
        p(z|x) is approximated using IS on log(p(x))
        """
        logpzx = self.log_p_z_given_x(z, recon_x, x, sample_size=sample_size)
        logqzx = torch.distributions.MultivariateNormal(
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
        beta_zero=0.3,
        tempering="fixed",
        model_type="mlp",
        input_dim=784,
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
        VAE.__init__(
            self, model_type=model_type, input_dim=input_dim, latent_dim=latent_dim
        )

        self.name = "HVAE"

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

        recon_x, z0, _, mu, log_var = self.vae_forward(x)
        gamma = torch.randn_like(z0, device=self.device)
        rho = gamma / self.beta_zero_sqrt
        z = z0
        beta_sqrt_old = self.beta_zero_sqrt

        recon_x = self.decode(z)

        for k in range(self.n_lf):

            # Perform leapfrog steps

            # Computes potential energy
            U = -self.log_p_xz(recon_x, x, z).sum()

            # Compute its gradient
            g = grad(U, z, create_graph=True)[0]

            # 1st leapfrog step
            rho_ = rho - (self.eps_lf / 2) * g
            z = z + self.eps_lf * rho_

            recon_x = self.decode(z)

            # 2nd leapfrog step
            U = -self.log_p_xz(recon_x, x, z).sum()
            g = grad(U, z, create_graph=True)[0]

            rho__ = rho_ - (self.eps_lf / 2) * g

            # tempering steps
            beta_sqrt = self._tempering(k)
            rho = (beta_sqrt / beta_sqrt_old) * rho__
            beta_sqrt_old = beta_sqrt

        return recon_x, z, z0, rho, gamma, mu, log_var

    def loss_function(self, recon_x, x, z0, zK, rhoK, gamma, mu, log_var):

        logpxz = self.log_p_xz(recon_x, x, zK)  # log p(x, z)
        logrhoK = self.normal.log_prob(rhoK)  # log p(\rho_K)
        logp = logpxz + logrhoK

        logqzx = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=torch.diag_embed(torch.exp(log_var))
        ).log_prob(
            z0
        )  # log(q(z|x))

        logrho0 = self.normal.log_prob(gamma)
        logq = logqzx + logrho0
        return -(logp - logq).mean()

    def log_p_x(self, x, sample_size=10):
        """
        Estimate log(p(x)) using importance sampling on q(z|x)
        """
        mu, log_var = self.encode(x.view(-1, self.input_dim))
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=self.device)
        Z = (mu + Eps * torch.exp(0.5 * log_var)).reshape(-1, self.latent_dim)
        Z0 = Z
        recon_X = self.decode(Z)

        gamma = torch.randn_like(Z, device=self.device)
        rho = gamma / self.beta_zero_sqrt
        rho0 = rho
        beta_sqrt_old = self.beta_zero_sqrt
        X_rep = x.repeat(sample_size, 1, 1, 1)

        for k in range(self.n_lf):

            U = self.hamiltonian(recon_X, X_rep, Z, rho)
            g = grad(U, Z, create_graph=True)[0]
            rho_ = rho - (self.eps_lf) * g

            Z = Z + self.eps_lf * rho

            recon_X = self.decode(Z)

            U = self.hamiltonian(recon_X, X_rep, Z, rho)
            g = grad(U, Z, create_graph=True)[0]

            rho__ = rho_ - (self.eps_lf / 2) * g

            # tempering
            beta_sqrt = self._tempering(k)
            rho = (beta_sqrt / beta_sqrt_old) * rho__
            beta_sqrt_old = beta_sqrt

        bce = F.binary_cross_entropy(
            recon_X, x.view(-1, self.input_dim).repeat(sample_size, 1), reduction="none"
        )

        # compute densities to recover p(x)
        logpxz = -bce.reshape(sample_size, -1, self.input_dim).sum(dim=2)  # log(p(x|z))

        logpz = self.log_z(Z).reshape(sample_size, -1)  # log(p(z))

        logrho0 = self.beta_zero_sqrt * self.normal.log_prob(rho0).reshape(sample_size, -1)  # log(p(rho0))
        logrho = self.normal.log_prob(rho).reshape(sample_size, -1)  # log(p(rho))

        logqzx = (
            torch.distributions.MultivariateNormal(
                loc=mu, covariance_matrix=torch.diag_embed(torch.exp(log_var))
            )
            .log_prob(Z0.reshape(sample_size, -1, self.latent_dim))
            .reshape(sample_size, -1)
        )  # log(q(z|x))

        logpx = (logpxz + logpz + logrho - logrho0 - logqzx).logsumexp(dim=0).mean(
            dim=0
        ) - torch.log(torch.Tensor([sample_size]).to(self.device))
        return logpx

    def hamiltonian(self, recon_x, x, z, rho, G=None, G_log_det=None):

        if self.name == "HVAE":
            return -self.log_p_xz(recon_x, x, z).sum()

        norm = (torch.solve(rho[:, :, None], G).solution[:, :, 0] * rho).sum()
        return -self.log_p_xz(recon_x, x, z).sum() + 0.5 * norm + 0.5 * G_log_det.sum()

    def _tempering(self, k):
        """Perform tempering step"""

        beta_k = (
            1 - (1 / self.beta_zero_sqrt) * k / self.n_lf ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k


class RHVAE(HVAE):
    def __init__(
        self,
        n_lf=3,
        eps_lf=0.01,
        beta_zero=0.3,
        tempering="fixed",
        model_type="mlp",
        input_dim=784,
        latent_dim=2,
    ):

        HVAE.__init__(
            self, n_lf, eps_lf, beta_zero, tempering, model_type, input_dim, latent_dim
        )

        self.name = "RHVAE"

    def forward(self, x):
        """
        Perform Riemann Hamiltionian Importance Sampling
        """

        recon_x, z0, _, mu, log_var = self.vae_forward(x)

        z = z0

        # Define a metric G(x) = \Sigma^{-1}(x)
        G = torch.diag_embed((-log_var).exp())
        G_log_det = torch.logdet(G)

        # gamma = torch.randn_like(z0, device=self.device)
        # rho = gamma / self.beta_zero_sqrt

        # Sample gamma
        gamma = torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(z), covariance_matrix=G
        ).sample()

        rho = gamma / self.beta_zero_sqrt

        beta_sqrt_old = self.beta_zero_sqrt

        for k in range(self.n_lf):

            # Perform leapfrog steps
            rho_ = self.leap_step_1(recon_x, x, z, rho, G, G_log_det)
            z = self.leap_step_2(recon_x, x, z, rho_, G, G_log_det)

            recon_x = self.decode(z)

            rho__ = self.leap_step_3(recon_x, x, z, rho_, G, G_log_det)

            rho = rho__

            # tempering steps
            beta_sqrt = self._tempering(k)
            rho = (beta_sqrt / beta_sqrt_old) * rho__
            beta_sqrt_old = beta_sqrt

        return recon_x, z, z0, rho, gamma, mu, log_var

    def log_p_x(self, x, sample_size=10):
        """
        Estimate log(p(x)) using importance sampling on q(z|x)
        """

        mu, log_var = self.encode(x.view(-1, self.input_dim))
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=self.device)
        Z = (mu + Eps * torch.exp(0.5 * log_var)).reshape(-1, self.latent_dim)
        Z0 = Z

        G = torch.diag_embed((-log_var).exp())
        G_rep = G.repeat(sample_size, 1, 1)
        G_log_det = G.logdet()
        G_log_det_rep = G_log_det.repeat(sample_size, 1, 1)

        recon_X = self.decode(Z)

        gamma = torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(Z), covariance_matrix=G_rep
        ).sample()

        rho = gamma / self.beta_zero_sqrt
        rho0 = rho

        beta_sqrt_old = self.beta_zero_sqrt
        X_rep = x.repeat(sample_size, 1, 1, 1)

        for k in range(self.n_lf):

            rho_ = self.leap_step_1(recon_X, X_rep, Z, rho, G_rep, G_log_det_rep)
            Z = self.leap_step_2(recon_X, X_rep, Z, rho_, G_rep, G_log_det_rep)

            recon_X = self.decode(Z)

            rho__ = self.leap_step_3(recon_X, X_rep, Z, rho_, G_rep, G_log_det_rep)

            # tempering
            beta_sqrt = self._tempering(k)
            rho = (beta_sqrt / beta_sqrt_old) * rho__
            beta_sqrt_old = beta_sqrt

        bce = F.binary_cross_entropy(
            recon_X, x.view(-1, self.input_dim).repeat(sample_size, 1), reduction="none"
        )

        # compute densities to recover p(x)
        logpxz = -bce.reshape(sample_size, -1, self.input_dim).sum(dim=2)  # log(p(x|z))

        logpz = self.log_z(Z).reshape(sample_size, -1)  # log(p(z))

        logrho0 = self.beta_zero_sqrt * (
            torch.distributions.MultivariateNormal(
                loc=torch.zeros_like(rho0), covariance_matrix=G_rep
            )
            .log_prob(rho0)
            .reshape(sample_size, -1)
        )
        # log(p(rho0))
        logrho = (
            torch.distributions.MultivariateNormal(
                loc=torch.zeros_like(rho), covariance_matrix=G_rep
            )
            .log_prob(rho)
            .reshape(sample_size, -1)
        )  # log(p(rho0))

        logqzx = (
            torch.distributions.MultivariateNormal(
                loc=mu, covariance_matrix=torch.diag_embed(torch.exp(log_var))
            )
            .log_prob(Z0.reshape(sample_size, -1, self.latent_dim))
            .reshape(sample_size, -1)
        )  # log(q(z|x))

        logpx = (logpxz + logpz + logrho - logrho0 - logqzx).logsumexp(dim=0).mean(
            dim=0
        ) - torch.log(torch.Tensor([sample_size]).to(self.device))

        return logpx

    def sample_img(self, z=None, n_samples=1, leap_step=True, leap_nbr=10):
        if z is None:
            z = self.normal.sample(sample_shape=(n_samples,)).to(self.device)

        else:
            n_samples = z.shape[0]

        z.requires_grad_(True)
        recon_x = self.decode(z)
        x = torch.distributions.Bernoulli(probs=recon_x).sample()

        _, log_var = self.encode(x)

        # Define a metric G(x) = \Sigma^{-1}(x)
        G = torch.diag_embed((-log_var).exp())
        G_log_det = torch.logdet(G)

        if leap_step:

            gamma = torch.distributions.MultivariateNormal(
                loc=torch.zeros_like(z), covariance_matrix=G
            ).sample()

            beta_sqrt_old = self.beta_zero_sqrt
            rho = gamma / self.beta_zero_sqrt

            for k in range(leap_nbr):

                # Perform leapfrog steps
                rho_ = self.leap_step_1(recon_x, x, z, rho, G, G_log_det)
                z = self.leap_step_2(recon_x, x, z, rho_, G, G_log_det)

                recon_x = self.decode(z)

                rho__ = self.leap_step_3(recon_x, x, z, rho_, G, G_log_det)

                rho = rho__

                # tempering steps
                beta_sqrt = self._tempering(k)
                rho = (beta_sqrt / beta_sqrt_old) * rho__
                beta_sqrt_old = beta_sqrt

        return recon_x

    def leap_step_1(self, recon_x, x, z, rho, G, G_log_det, steps=3):
        """
        Resolves eq.16 from Girolami using fixed point iterations
        """

        def f_(rho):
            H = self.hamiltonian(recon_x, x, z, rho, G, G_log_det)
            gz = grad(H, z, retain_graph=True)[0]
            return rho - 0.5 * self.eps_lf * gz

        rho_ = rho.clone()
        for _ in range(steps):
            rho_ = f_(rho_)
        return rho_

    def leap_step_2(self, recon_x, x, z, rho, G, G_log_det, steps=3):
        """
        Resolves eq.17 from Girolami using fixed point iterations
        """
        H0 = self.hamiltonian(recon_x, x, z, rho, G, G_log_det)
        grho_0 = grad(H0, rho)[0]

        def f_(z):
            H = self.hamiltonian(recon_x, x, z, rho, G, G_log_det)
            grho = grad(H, rho, retain_graph=True)[0]
            return z + 0.5 * self.eps_lf * (grho_0 + grho)

        z_ = z.clone()
        for _ in range(steps):
            z_ = f_(z_)
        return z_

    def leap_step_3(self, recon_x, x, z, rho, G, G_log_det, steps=3):
        H = self.hamiltonian(recon_x, x, z, rho, G, G_log_det)
        gz = grad(H, z, create_graph=True)[0]
        return rho - 0.5 * self.eps_lf * gz


class AdaRHVAE(RHVAE):
    def __init__(
        self,
        n_lf=3,
        eps_lf=0.01,
        beta_zero=0.3,
        metric="sigma",
        tempering="fixed",
        model_type="mlp",
        input_dim=784,
        latent_dim=2,
    ):

        HVAE.__init__(
            self, n_lf, eps_lf, beta_zero, tempering, model_type, input_dim, latent_dim
        )

        self.name = "RHVAE"

        assert metric in [
            "sigma",
            "jacobian",
            "fisher",
        ], f"The metric {metric} is not handled by the RHVAE"

        self.metric = metric

    def forward(self, x):
        """
        Perform Riemann Hamiltionian Importance Sampling
        """

        recon_x, z0, _, mu, log_var = self.vae_forward(x)

        z = z0

        if self.metric == "jacobian":
            # Define metric G(z) = Jac(g(z))
            J_bis = self.jacobian_bis(recon_x, z)
            G = torch.transpose(J_bis, 1, 2) @ J_bis + 1e-7 * torch.eye(self.latent_dim).to(self.device)
            G_log_det = torch.logdet(G)

        elif self.metric == "fisher":
            G = self.fisher(recon_x, z, n_samples=100)
            G_log_det = torch.logdet(G)

        elif self.metric == "sigma":
            # Define a metric G(x) = \Sigma^{-1}(x)
            G = torch.diag_embed((-log_var).exp())
            G_log_det = torch.logdet(G)

        gamma = torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(z), covariance_matrix=G
        ).sample()

        rho = gamma / self.beta_zero_sqrt

        beta_sqrt_old = self.beta_zero_sqrt

        recon_x = self.decode(z)

        for k in range(self.n_lf):

            # Perform leapfrog steps
            rho_ = self.leap_step_1(recon_x, x, z, rho, G, G_log_det)
            z = self.leap_step_2(recon_x, x, z, rho_, G, G_log_det)

            recon_x = self.decode(z)

            if self.metric == "jacobian":
                J_bis = self.jacobian_bis(recon_x, z)
                G = torch.transpose(J_bis, 1, 2) @ J_bis + 1e-7 * torch.eye(self.latent_dim).to(self.device)
                G_log_det = torch.logdet(G)

            elif self.metric == "sigma":
                G = torch.diag_embed((-log_var).exp())
                G_log_det = torch.logdet(G)

            elif self.metric == "fisher":
                G = self.fisher(recon_x, z, n_samples=100)
                G_log_det = torch.logdet(G)

            rho__ = self.leap_step_3(recon_x, x, z, rho_, G, G_log_det)

            rho = rho__
            # tempering steps
            beta_sqrt = self._tempering(k)
            rho = (beta_sqrt / beta_sqrt_old) * rho__
            beta_sqrt_old = beta_sqrt

        return recon_x, z, z0, rho, gamma, mu, log_var

    def log_p_x(self, x, sample_size=10):
        """
        Estimate log(p(x)) using importance sampling on q(z|x)
        """

        mu, log_var = self.encode(x.view(-1, self.input_dim))
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=self.device)
        Z = (mu + Eps * torch.exp(0.5 * log_var)).reshape(-1, self.latent_dim)
        Z0 = Z

        recon_X = self.decode(Z)

        if self.metric == "jacobian":
            J_rep = self.jacobian_bis(
                recon_X.reshape(-1, self.input_dim), Z.reshape(-1, self.latent_dim)
            )
            G_rep = torch.transpose(J_rep, 1, 2) @ J_rep + 1e-7 * torch.eye(self.latent_dim).to(self.device)
            G_log_det_rep = torch.logdet(G_rep)
            G_rep0 = G_rep

        elif self.metric == "sigma":
            G_rep = torch.diag_embed((-log_var).exp())
            G_log_det_rep = torch.logdet(G)
            G_rep0 = G_rep

        elif self.metric == "fisher":
            G_rep = self.fisher(recon_X, Z, n_samples=100)
            G_log_det_rep = torch.logdet(G_rep)
            G_rep0 = G_rep

        gamma = torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(Z), covariance_matrix=G_rep
        ).sample()

        rho = gamma / self.beta_zero_sqrt

        rho0 = rho

        beta_sqrt_old = self.beta_zero_sqrt

        X_rep = x.repeat(sample_size, 1, 1, 1)

        for k in range(self.n_lf):

            rho_ = self.leap_step_1(recon_X, X_rep, Z, rho, G_rep, G_log_det_rep)
            Z = self.leap_step_2(recon_X, X_rep, Z, rho_, G_rep, G_log_det_rep)

            recon_X = self.decode(Z)

            if self.metric == "jacobian":
                J_rep = self.jacobian_bis(
                    recon_X.reshape(-1, self.input_dim), Z.reshape(-1, self.latent_dim)
                )
                G_rep = torch.transpose(J_rep, 1, 2) @ J_rep + 1e-7 * torch.eye(self.latent_dim).to(self.device)
                G_log_det_rep = torch.logdet(G_rep)

            elif self.metric == "fisher":
                G_rep = self.fisher(recon_X, Z, n_samples=100)
                G_log_det_rep = torch.logdet(G_rep)

            rho__ = self.leap_step_3(recon_X, X_rep, Z, rho_, G_rep, G_log_det_rep)

            # tempering
            beta_sqrt = self._tempering(k)
            rho = (beta_sqrt / beta_sqrt_old) * rho__
            beta_sqrt_old = beta_sqrt

        bce = F.binary_cross_entropy(
            recon_X, x.view(-1, self.input_dim).repeat(sample_size, 1), reduction="none"
        )

        # compute densities to recover p(x)
        logpxz = -bce.reshape(sample_size, -1, self.input_dim).sum(dim=2)  # log(p(x|z))

        logpz = self.log_z(Z).reshape(sample_size, -1)  # log(p(z))

        logrho0 = self.beta_zero_sqrt * (
            torch.distributions.MultivariateNormal(
                loc=torch.zeros_like(rho0), covariance_matrix=G_rep0
            )
            .log_prob(rho0)
            .reshape(sample_size, -1)
        )
        # log(p(rho0))
        logrho = (
            torch.distributions.MultivariateNormal(
                loc=torch.zeros_like(rho), covariance_matrix=G_rep
            )
            .log_prob(rho)
            .reshape(sample_size, -1)
        )  # log(p(rho0))
        logqzx = (
            torch.distributions.MultivariateNormal(
                loc=mu, covariance_matrix=torch.diag_embed(torch.exp(log_var))
            )
            .log_prob(Z0.reshape(sample_size, -1, self.latent_dim))
            .reshape(sample_size, -1)
        )  # log(q(z|x))

        logpx = (logpxz + logpz + logrho - logrho0 - logqzx).logsumexp(dim=0).mean(
            dim=0
        ) - torch.log(torch.Tensor([sample_size]).to(self.device))

        return logpx

    def fisher(self, recon_x, z, n_samples=1):
        """
        Compute estimate of the Fisher information matrix using MC
        """
        z_ = z.clone().detach()
        noutputs = recon_x.shape[1]
        z_ = z_.unsqueeze(1)  # b, 1 ,in_dim
        n = z_.size()[0]
        z_ = z_.repeat(1, noutputs, 1)  # b, out_dim, in_dim
        z_.requires_grad_(True)
        recon_x_ = self.decode(z_)

        input_val = (
            torch.eye(noutputs)
            .reshape(1, noutputs, noutputs)
            .repeat(n, 1, 1)
            .to(self.device)
        )

        jac = grad(recon_x_, z_, grad_outputs=input_val)[0]

        # Estimate susing MC sampling
        x_samples = (
            torch.distributions.Bernoulli(probs=recon_x)
            .sample(sample_shape=(n_samples,))
            .transpose(0, 1)
        )

        # print(x_samples.shape, jac.shape, recon_x.shape)

        # Compute derivative of the log proba \partial log p(x|z)
        d_z = torch.transpose(jac, 1, 2) @ (
            -(1 - x_samples) / (1 - recon_x.unsqueeze(1))
            + x_samples / recon_x.unsqueeze(1)
        ).transpose(1, 2)
        d_z = d_z.transpose(1, 2).unsqueeze(-1)

        # Compute [\nabla \partial log p(x|x) \nabla \partial log p(x|z)^{\top}]
        d_z_mat = d_z @ torch.transpose(d_z, 2, 3)
        # print(d_z_mat.shape)

        # Compute expectation
        fisher = d_z_mat.mean(dim=1)

        return fisher + 1e-7 * torch.eye(self.latent_dim).to(self.device)

    def jacobian(self, recon_x, z, eps=0.00001):
        """
        Compute the Jacobian matrix of the output of neural net w.r.t. the inputs
        using finite differences scheme

        Inputs:
        -------
        recon_x (Tensor): The output of the network [Batch_size, output_size]
        z (Tensor): The input [Batch_size, latent_dim]
        eps (float): The precision used in the finite difference scheme
        """

        _, n = z.shape
        jacob = list()
        for i in range(n):
            z_eps = z.clone()
            z_eps[:, i] += eps
            dnet_i_dz = (self.decode(z_eps) - recon_x) / eps
            jacob.append(dnet_i_dz)

        jacob = torch.stack(jacob, dim=2)
        return jacob

    def jacobian_bis(self, recon_x, z):
        """
        Compute the Jacobian matrix of the output of neural net w.r.t. the inputs
        using PyTorch autograd

        Inputs:
        -------
        recon_x (Tensor): The output of the network [Batch_size, output_size]
        z (Tensor): The input [Batch_size, latent_dim]
        """
        z_ = z.clone().detach()

        noutputs = recon_x.shape[1]
        z_ = z_.unsqueeze(1)  # b, 1 ,in_dim
        n = z_.size()[0]
        z_ = z_.repeat(1, noutputs, 1)  # b, out_dim, in_dim
        z_.requires_grad_(True)
        recon_x_ = self.decode(z_)

        input_val = (
            torch.eye(noutputs)
            .reshape(1, noutputs, noutputs)
            .repeat(n, 1, 1)
            .to(self.device)
        )

        jac = grad(recon_x_, z_, grad_outputs=input_val)[0]

        return jac 

    def sample_img(self, z=None, n_samples=1, leap_step=True, leap_nbr=10):
        if z is None:
            z = self.normal.sample(sample_shape=(n_samples,)).to(self.device)

        else:
            n_samples = z.shape[0]

        z.requires_grad_(True)
        recon_x = self.decode(z)
        x = torch.distributions.Bernoulli(probs=recon_x).sample()

        if self.metric == "jacobian":
            J = self.jacobian_bis(recon_x, z)
            G = torch.transpose(J, 1, 2) @ J
            G_log_det = torch.logdet(G)

        elif self.metric == "fisher":
            G = self.fisher(recon_x, z, n_samples=100)
            G_log_det = torch.logdet(G)

        elif self.metric == "sigma":
            _, log_var = self.encode(x)
            # Define a metric G(x) = \Sigma^{-1}(x)
            G = torch.diag_embed((-log_var).exp())
            G_log_det = torch.logdet(G)

        if leap_step:

            gamma = torch.distributions.MultivariateNormal(
                loc=torch.zeros_like(z), covariance_matrix=G
            ).sample()
            beta_sqrt_old = self.beta_zero_sqrt
            rho = gamma / self.beta_zero_sqrt

            for k in range(leap_nbr):

                # Perform leapfrog steps
                rho_ = self.leap_step_1(recon_x, x, z, rho, G, G_log_det)
                z = self.leap_step_2(recon_x, x, z, rho_, G, G_log_det)

                recon_x = self.decode(z)

                if self.metric == "jacobian":
                    recon_x = self.decode(z)
                    J_bis = self.jacobian(recon_x, z)
                    G = torch.transpose(J_bis, 1, 2) @ J_bis
                    G_log_det = torch.logdet(G)

                elif self.metric == "fisher":
                    recon_x = self.decode(z)
                    x = torch.distributions.Bernoulli(probs=recon_x).sample()
                    G = self.fisher(recon_x, z, n_samples=100)
                    G_log_det = torch.logdet(G)

                rho__ = self.leap_step_3(recon_x, x, z, rho_, G, G_log_det)

                rho = rho__

                # tempering steps
                beta_sqrt = self._tempering(k)
                rho = (beta_sqrt / beta_sqrt_old) * rho__
                beta_sqrt_old = beta_sqrt

        return recon_x
