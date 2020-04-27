---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: phd
    language: python
    name: phd
---

```python
import torch
import matplotlib.pyplot as plt
%matplotlib inline
%load_ext autoreload
%autoreload 2
from tqdm import*
from vae.samplers.mcmc import HM, HMC
```

```python
mu = torch.zeros(2) + 5
Sigma = torch.diag(torch.Tensor([1, 100]))
gaussian = torch.distributions.MultivariateNormal(mu, Sigma)
log_gauss = gaussian.log_prob
x = torch.randn(10000,2) + torch.Tensor([0,60])
```

```python
def banana(x):
    """x.shape = [n, d]"""
    y = torch.zeros_like(x)
    y[:,1] = 0.3*(x[:,0]**2 - 200)
    return gaussian.log_prob(x + y)
```

```python
y = x[banana(x) > -1e1]
plt.scatter(x[:,0], x[:,1], c=banana(x).exp())
plt.show()
```

```python
g = gaussian.sample(sample_shape=torch.Size([10000])).numpy()
plt.scatter(g[:, 0], g[:, 1])
```

```python
hm = HM(chain_size=5000, dim=2, seed=42)

mcmc = hm.sample(log_gauss, sigma=2)
plt.scatter(g[:,0], g[:,1], alpha=0.5)
plt.plot(mcmc[:,0].numpy(), mcmc[:,1].numpy(), c="red", alpha=0.7)
plt.show()
```

```python
hmc = HMC(chain_size=100, dim=2, seed=42)

M = torch.eye(2)
x0 = torch.tensor([20, 10])

mcmc = hmc.sample(20, log_gauss, M=M, x0=x0)[0]

plt.scatter(g[:,0], g[:,1], alpha=0.5)
plt.plot(mcmc[:,0].numpy(), mcmc[:,1].numpy(), c="red", alpha=0.7)
plt.show()
```

```python
def G(x):
    return torch.inverse(Sigma)

rhmc = HMC(chain_size=100, dim=2, seed=42)
x0 = torch.tensor([20, 10])
mcmc = rhmc.sample(20, log_gauss, G=G, x0=x0)[0]

plt.scatter(g[:,0], g[:,1], alpha=0.5)
plt.plot(mcmc[:,0].numpy(), mcmc[:,1].numpy(), c="red", alpha=0.7)
plt.show()
```
