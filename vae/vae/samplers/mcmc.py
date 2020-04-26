import torch
import numpy as np
from torch.autograd import grad
from .base import BaseMCMCSampler
from tqdm import tqdm_notebook


class HM(BaseMCMCSampler):
    """

    """

    def __init__(self, chain_size, dim, seed=None):
        BaseMCMCSampler.__init__(self, chain_size, dim, seed)
        self.acc = 0

    def sample(self, log_proba, sigma, x0=None, display_progress=True):

        x = torch.zeros(self.chain_size, self.dim)

        if x0 is not None:
            assert isinstance(x0, torch.Tensor), "The starting point is not a tensor"
            assert (
                x0.size()[-1] == self.dim
            ), f"Starting point lives in dimension {x0.size()[-1]} while dimension {self.dim} is requested"

            x[0] = x0

        it = (
            tqdm_notebook(range(1, self.chain_size))
            if display_progress
            else range(1, self.chain_size)
        )

        for i in it:
            x[i] = x[i - 1] + sigma * torch.randn(self.dim)
            log_alpha = log_proba(x[i][None]) - log_proba(x[i - 1][None])

            if torch.rand(1).log() > log_alpha:
                x[i] = x[i - 1]
            else:
                self.acc += 1

        if display_progress:
            print("Acceptance ratio: {:03.2f}".format(self.acc / self.chain_size))

        return x


class HMC(BaseMCMCSampler):
    def __init__(self, chain_size, dim, seed=None):
        BaseMCMCSampler.__init__(self, chain_size, dim, seed)
        self.acc = 0

    def sample(
        self,
        n_lf,
        log_proba,
        M=None,
        x0=None,
        eps=0.1,
        acc_temp=1,
        display_progress=True,
        skip_acceptance=False,
        full_trajectory=False,
    ):
        """
        Hamiltonian MC sampling

        Inputs:
        -------

        n_lf (int): number of leapfrog steps per iteration
        log_prob (function): function taking [chain_size, dimension] arrays and returs a [chain_size] log_probability
        x0 (Tensor): starting point
        M (Tensor): Momentum Matrix
        eps (float): leapfrog step size
        T (float): acceptance temperature
        display_progess (Bool): see evolution
        skip_acceptance (Bool): skip acceptance or not
        full_trajectory (Bool): - if True, returns the concatenation of all leapfrog steps,
                                - if False, returns only the intermediate points

        """
        if x0 is not None:
            assert isinstance(
                x0, torch.Tensor
            ), f"The starting point is not a tensor got {type(x0)} type"
            assert (
                x0.size()[-1] == self.dim
            ), f"Starting point lives in dimension {x0.size()[-1]} while dimension {self.dim} is requested"

        else:
            x0 = torch.zeros(self.dim)

        if M is None:
            M = torch.eye(self.dim)

        assert isinstance(
            M, torch.Tensor
        ), f"Momentum matrix must be a tensor got {type(M)} type"
        assert tuple(M.size()) == (
            self.dim,
            self.dim,
        ), f"Momentum matrix dimension of [{M.size()[0]}x{M.size()[1]}] while [{self.dim}x{self.dim}] is requested"

        # Compute inverse and logdet to avoid repeated computations in the loop
        M_inv = torch.inverse(M)
        M_log_det = torch.logdet(M)

        if full_trajectory:
            x = torch.zeros(self.chain_size * n_lf, self.dim)
            p = torch.zeros(self.chain_size * n_lf, self.dim)

        else:
            x = torch.zeros(self.chain_size + 1, self.dim)
            p = torch.zeros(self.chain_size + 1, self.dim)
            x[0] = x0
        x_init = x0

        it = (
            tqdm_notebook(range(0, self.chain_size))
            if display_progress
            else range(0, self.chain_size)
        )

        for i in it:

            x_lf, p_lf = self.__leapfrog(n_lf, log_proba, M, M_inv, x_init)
            if not skip_acceptance:
                H_start = self.__hamiltonian(
                    x_lf[0], p_lf[0], M_inv, M_log_det, log_proba
                )
                H_end = self.__hamiltonian(
                    x_lf[-1], p_lf[-1], M_inv, M_log_det, log_proba
                )

            if skip_acceptance or torch.rand(1).log() < (-H_end + H_start) / acc_temp:
                if full_trajectory:
                    x[i * n_lf : (i + 1) * n_lf] = x_lf[1:]
                    p[i * n_lf : (i + 1) * n_lf] = p_lf[1:]
                    x_init = x[(i + 1) * n_lf - 1]
                else:
                    x[i + 1], p[i + 1] = x_lf[-1], p_lf[-1]
                    x_init = x[i + 1]
                self.acc += 1

            else:
                if full_trajectory:
                    x[i * n_lf : (i + 1) * n_lf] = x[i * n_lf - 1]
                    p[i * n_lf : (i + 1) * n_lf] = p[i * n_lf - 1]
                else:
                    x[i + 1] = x[i]
                    p[i + 1] = p[i]

        if display_progress:
            print("Acceptance rate: {:03.2f}".format(self.acc / self.chain_size))

        return x, p

    def __leapfrog(
        self,
        n_lf,
        log_proba,
        M,
        M_inv,
        x0,
        eps=0.1,
        display_progress=False,
        only_final=False,
    ):
        """
        Performs leapfrog steps

        Inputs:
        -------

        n_lf (int): length of the chain
        log_prob (function): function taking [chain_size, dimension] arrays and returns a [chain_size] log_probability
        momentum matrix (tensor): inverse of momentum matrix
        x0 (Tensor): starting point
        eps (float): leapfrog step size
        display_progess (Bool): see evolution
        only_final (Bool): if True, only returns thre last point of trajectory

        Note:
        -----
        The sampling of the momentum variable is performed with this function
        """

        x = torch.zeros(n_lf + 1, self.dim)
        p = torch.zeros(n_lf + 1, self.dim)
        x[0] = x0

        # Sample momentum
        p[0] = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.dim), covariance_matrix=M
        ).sample()

        it = (
            tqdm_notebook(range(1, n_lf + 1))
            if display_progress
            else range(1, n_lf + 1)
        )
        for i in it:
            x_ = x[i - 1].clone()
            x_.requires_grad_(True)
            p[i] = p[i - 1] + 0.5 * eps * grad(log_proba(x_[None]), x_)[0]
            x[i] = x_.detach_() + eps * M_inv @ p[i]
            x_ = x[i].clone()
            x_.requires_grad_(True)
            p[i] = p[i] + 0.5 * eps * grad(log_proba(x_[None]), x_)[0]

        if only_final:
            return x[-1], p[-1]

        return x, p

    def __hamiltonian(self, x, p, M_inv, M_log_det, log_proba):
        """
        Computes the Hamiltonian

        Inputs:
        -------

        x (Tensor): position tensor of size (chain_size)
        p (Tensor): momentum tensor of sier (chain_size)
        log_proba (function): function taking [chain_size, dimension] arrays and returns a [chain_size] log_probability
        """

        return (
            -log_proba(x[None])
            + 0.5 * M_log_det
            + 0.5 * p.T @ M_inv @ p
            + 0.5 * self.dim * torch.Tensor([2 * np.pi]).log()
        )
