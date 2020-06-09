# Riemannian Hamiltonian Variational Auto-Encoder with a Learned Metric

This repository is the official implementation of Riemannian Hamiltonian Variational Auto-Encoder with a Learned Metric


## Requirements

To install requirements run:

```setup
pip install -r requirements.txt
```

## Running tests

### Experiments description

Four experiments are proposed:

- **metric** (sec 4.2 Metric Computation): This experiment consists in training 3 RHVAE with 3 different fixed metric temperatures equal to 0.6, 0.8 and 1 on 3 classes of 50 samples each of the MNIST dataset. Metrics' magnitude and orientation are then displayed for each temperature. Experiment results are available in `experiments_plots/metric_MM_DD_YYYY_hh_mm_ss` folder.
- **reconstruction** (sec 4.3 Auto-encoder): This experiments aims at comparing the reconstruction and log-likelihood performances of classic VAE, HVAE and RHVAE trained on 80% of a dataset composed by 3 classes of 50 samples randomly selected. We propose to display samples generated by each model at the end of the 500 epochs of training as a complement to the paper. 90 MCMC steps were considered for the RHVAE. Experiment results are available in `experiments_plots/reconstruction_MM_DD_YYYY_hh_mm_ss` folder.
- **generation_5** (sec 4.4 Generation (1 class)): In this experiment we compare the models in terms of generation. Each of them is trained for 1000 epochs on 160 samples of 1 class of the MNIST dataset (5's). 90 MCMC steps were considered for the RHVAE. Experiment results are available in `experiments_plots/generation_5_MM_DD_YYYY_hh_mm_ss` folder.
- **generation_7_classes** (sec 4.4 Generation (7 classes)): This experiment consists in training a RHVAE with a fixed metric temperature equal to 0.6 on 80% on 7 classes of 50 samples randomly selected. Generation was performed with 150 MCMC steps. Experiment results are available in `experiments_plots/generation_7_classes_MM_DD_YYYY_hh_mm_ss` folder

Unless stated otherwise above, all parameters were set as follows:

- **VAE**: latent dimension (2), input dimension (784)
- **HVAE**: latent dimension (2), input dimension (784), number of leapfrog step (10), leapfrog step size (0.01), initial tempering factor (0.3)
- **RHVAE**: latent dimension (2), input dimension (784), number of leapfrog step (10), leapfrog step size (0.1), initial tempering factor (0.3), metric temperature (fixed and equal to 1), metric regularization (fixed and equal to 0.1).

### Testing models

To test pre-trained models and reproduce the results presented in the paper run:

```bash
./experiments.sh
```

This  will load pre-trained models and perform all the experiments described above.


To perform the end-to-end procedure (i.e. with training) run:

```bash
./full_experiments.sh
```

This will train the VAE, HVAE and RHVAE models with the same parameters as described in the paper and above, store them in the folder `trained_models` and run all the experiments described above.

The results for each experiment can be found in `experiments_plots` in a folder with a unique identifier `EXPERIMENT_NAME_MM_DD_YYY_hh_mm_ss`.


## Pre-trained Models

You can find pretrained models, loaders and metrics in `pretrained_models`. These models were used to produce te results presented in the paper.


## Training

To train the model(s) with the same data as presented in the paper, commands should have the following form:

```train
python train.py -n <model_name> -lt <path_to_train_loader> -lv <path_to_test_loader> (-params <model_parameters>)
```

All arguments can be found in `train.py`
Models are trained on CPU.

### Example 1

To train a VAE with default parameters run:

```bash
python train.py -n vae -lt 'pretrained_models/generation_5/train_generation_5_loader_generation_5_final' -lv 'pretrained_models/generation_5/test_generation_5_loader_generation_5_final' -ne 1000'
```

#### Example 2

To train a RHVAE model on the data used for the experiment in sec 4.2 with 150 epochs, 10 leapfrog step, a leapfrog step size of 0.01, metric temperature of 1 and regularization factor of 0.1 run:

```bash
python train.py -n rhvae -lt 'pretrained_models/metric/train_loader_metric_final' -lv 'pretrained_models/metric/test_loader_metric_final' -ne 150 -n_lf 10 -eps_lf 0.01 -temp 1 -reg 0.1-spec 'T1'

```

## Short file description

- `train.py`: This is the main file to train the models.
- `eval.py`: This is the main file to run the experiments on the models.
- `plotting.py`: This file contains all the methods to plot the results for each experiment.
- `models/vae.py`: This file contains the models as classes (VAE, HVAE & RHVAE)
- `trainers/trainer.py`: This file contains the ModelTrainer class which stores the metrics and is used to train the model.
- `utils.py`: This file contains a class Digits used to instantiate the Data loaders with MNIST dataset, a method `create_metric` to recover the metric from a trained model and `load_model` used to facilitate model loading.

## Short folder description

- `experiments_plots`: After each experiment, the plots are stored in this folder with a unique identifier of the form `EXPERIMENT_NAME_MM_DD_YYY_hh_mm_ss`.
- `trained_models`: At the end of the training process, the models are stored in this folder.
- `pretrained_models`: This model contains pre-trained models used to perform the experiments presented in the paper. Each experiment has its own folder `EXPERIMENT_NAME` where models and loaders are stored.
