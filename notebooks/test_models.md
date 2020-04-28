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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import random


%matplotlib inline
%load_ext autoreload
%autoreload 2

from vae.models.vae import VAE, HVAE, RHVAE
from vae.trainers.trainer import ModelTrainer

```

```python
torch.manual_seed(1)

bs= 128

def random_binarize(img):
    return (torch.rand_like(img) < img).type(torch.float)

transforms_stack = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(random_binarize),
])

train = datasets.MNIST(root='../data/',
                       train=True,
                       transform=transforms_stack,
                       download=False)

test = datasets.MNIST(root='../data/',
                    train=False,
                    transform=transforms_stack,
                    download=False) 




# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=bs, shuffle=False)
```

```python
vae = VAE()
hvae = HVAE()
rhvae = RHVAE()
if torch.cuda.is_available():
    vae.cuda()
    hvae.cuda()
    rhvae.cuda()

```

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader.dataset.data.shape[0]

trainer = ModelTrainer(rhvae, train_loader, test_loader, n_epochs=30)

trainer.train()
```

```python
rhvae.name
```

```python
with torch.no_grad():
    z = torch.randn(64, 2).to(device)
    sample = hvae.decode(z)
    sample = torch.reshape(sample, (64, 28, 28))
plt.matshow(sample.cpu().numpy()[0])
```

```python
8 * 16
```

```python

```
