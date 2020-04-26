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

from vae.models.vae import VAE
from vae.trainers.trainer import ModelTrainer

```

```python
torch.manual_seed(1)

bs= 128

train = datasets.MNIST(root='../data/',
                       train=True,
                       transform=transforms.ToTensor(),
                       download=False)

test = datasets.MNIST(root='../data/',
                    train=False,
                    transform=transforms.ToTensor(),
                    download=False) 

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=bs, shuffle=False)
```

```python
vae = VAE()
if torch.cuda.is_available():
    vae.cuda()
```

```python
torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader.dataset.data.shape[0]
```

```python
trainer = ModelTrainer(vae, train_loader, test_loader, n_epochs=15)
trainer.train()
```

```python
60000 / 128
```

```python
with torch.no_grad():
    z = torch.randn(64, 2)
    sample = vae.decode(z)
    sample = torch.reshape(sample, (64, 28, 28))
plt.matshow(sample.cpu().numpy()[0])
```

```python

```
