# Riemannian Hamiltonian Variational Auto-Encoder with a Learned Metric

This repository is the official implementation of Riemannian Hamiltonian Variational Auto-Encoder with a Learned Metric


## Requirements

To install requirements run:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) with the same data sa presented in the paper commands should have the following form:

```train
python train.py -n <model_name> -lt <path_to_train_loader> -lv <path_to_test_loader> (-params <model_parameters>)
```

Models were trained on CPU.

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

## Running tests

Four experiments are proposed:

- metric (sec 4.2 Metric Computation)
- reconstruction (sec 4.3 Auto-encoder)
- generation_5 (sec 4.4 Generation (1 class))
- generation_7_classes (sec 4.4 Generation (7 classes)).

To test pretrained models and reproduce the results presented in the paper run:

```bash
./experiments.sh
```

To perform the end-to-end procedure (i.e. with training) run:

```bash
./full_experiments.sh
```

This will train the VAE, HVAE and RHVAE models needed with the same parameters as described in the paper and store them and the metrics in the folder `trained_models`. Then, the models are used to run the same experiments as mentioned above. 

The results for each experiment can be found in `experiments_plots` in a folder with a unique identifier `EXPERIMENT_NAME_MM_DD_YYY_hh_mm_ss`.


## Pre-trained Models

You can find pretrained models, loaders and metrics in `pretrained_models`.

## Short file description

- `train.py`: This is the main file to train the models.
- `eval.py`: This is the main file to run experiments on the models.
- `plotting.py`: This file contains all the methods to plot the results for each experiment.
- `models/vae.py`: This file contains the models as classes (VAE, HVAE & RHVAE)
- `trainers/trainer.py`: This file contains the ModelTrainer class which stores the metrics and is used to train the model.
- `utils.py`: This file contains a class Digits used to instantiate the Data loaders with MNIST data, a method `create_metric` to build the metric from a trained model and `load_model` used to facilitate model loading.

## Short folder description

- `experiments_plots`: After each experiment, the plots are stored in this folder with a unique identifier of the form `EXPERIMENT_NAME_MM_DD_YYY_hh_mm_ss`.
- `trained_models`: At the end of the training process, the models are stored in this folder.
- `pretrained_models`: This model contains pre-trained models used to perform the experiments presented in the paper. Each experiment has its own folder `EXPERIMENT_NAME` where models and loaders are stored.

