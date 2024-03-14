import optuna
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import click
import os.path as osp
import mlflow

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pytorch_lightning.loggers import TensorBoardLogger

from networks.SimpleClassifier import SimpleClassifier
from networks.DataModule import DataModule
from networks.engine import get_freer_gpu

from compute_labels import compute_atr, compute_fisher, compute_background

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

"""
Validate dense neural network: Validation with hyperparameters search.
"""
# Data directory.
datadir = "../data/"


def _compute_morgan(smile):
    """
    Function to compute morgan fingerprints for a list of smiles.
    """
    molecule = Chem.MolFromSmiles(smile)
    fp_object = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
    morgan_fp = np.zeros((0, ))
    DataStructs.ConvertToNumpyArray(fp_object, morgan_fp)
    
    return morgan_fp


def train_val_split(fingerprints, labels):
    """
    Get train and validation splits.
    """
    # Get train and validation splits.
    train_index = np.load("../data/splits/train_indices.npy", allow_pickle=False)
    val_index = np.load("../data/splits/val_indices.npy", allow_pickle=False)

    X_train, y_train = fingerprints[train_index], labels.loc[train_index]
    X_val, y_val = fingerprints[val_index], labels.loc[val_index]

    return X_train, y_train, X_val, y_val


def get_samples_weights(train_labels):
    """
    Compute class weights for WeightedRandomSampler.
    """
    class_samples_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1. / class_samples_count
    samples_weight = np.array([weight[t] for t in train_labels])
    samples_weight = torch.from_numpy(samples_weight)
    
    return samples_weight


def suggest_hyperparameters(trial: optuna.trial.Trial) -> list:
    """
    Suggest hyperparameters for optuna search.
    """
    # Obtain the learning rate on a logarithmic scale.
    learning_rate = trial.suggest_float("lr", low=1e-4, high=1e-1, log=True)
    # Obtain the dropout ratio.
    dropout = trial.suggest_float("dropout", low=0.0, high=0.8, step=0.1)
    # Obtain the number of layers.
    n_layers = trial.suggest_int("num_layers", low=2, high=5)
    # Obtain the number of units for the first hidden layer.
    n_units = trial.suggest_int("num_units", low=512, high=2048, step=1)

    print(f"Suggested hyperparameters: \n{(trial.params)}")
    return trial.params


def objective(trial: optuna.trial.Trial, expname:str, training_dataloader, validation_dataloader, devices:int) -> float:
    """
    Search for optimal set of hyperparameters
    """
    # Get hyperparameters.
    hyperparameters = suggest_hyperparameters(trial)

    with mlflow.start_run(run_name=f"{trial.number}") as run:
        mlflow.log_params(trial.params)

        # Instanciate the model.
        model = SimpleClassifier(2048, hyperparameters)
        # Initialize trainer.
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="gpu",
            devices=[devices],
            logger=False,
            callbacks=[
                EarlyStopping(monitor="val_mcc", mode="max"),
                ModelCheckpoint(monitor=None, filename=f"{expname}")
                ]
        )

        trainer.fit(model, train_dataloaders=training_dataloader, val_dataloaders=validation_dataloader)

        # Logging.
        mlflow.log_metric(f"val_loss", trainer.callback_metrics["val_loss"].item())
        mlflow.log_metric(f"val_mcc", trainer.callback_metrics["val_mcc"].item())

    return trainer.callback_metrics["val_mcc"].item()


@click.command()
@click.option("-d", "--dtype", required=True, help="metric to be used: atr/nar/fisher.")
def Main(dtype:str):
    """
    Perform hyperparameter search on validation set.
    """
    # Load data.
    print("\n Loading data ...")

    smiles = pd.read_csv(osp.join(datadir, "bioactivity_matrix.csv")).smiles.values
    fingerprints = np.array([_compute_morgan(i) for i in tqdm(df)], dtype=float)

    print("\nDataset specs: ")
    print("\t# Compound:", fingerprints.shape[0])
    print("\t# features:", fingerprints.shape[1])

    # Select load dataset according to datatype and select correct thresholds.
    if dtype == "atr":
        interference_metric = pd.read_csv(osp.join(datadir, "atr_values.csv"))
        thresholds = [0.9, 1.0, 3.0, 5.0]

    elif dtype == "fisher":
        interference_metric = pd.read_csv(osp.join(datadir, "p_values.csv"))
        thresholds = [0.35, 0.17, 0.07, 0.01]

    elif dtype == "back":
        interference_metric = pd.read_csv(osp.join(datadir, "nar_values.csv"))
        thresholds = [0.03, 0.04, 0.07, 0.1]

    # Get train and test data.
    X_train, labels_train, X_val, labels_val = train_val_split(fingerprints, interference_metric)

    # Run a study for each threshold.
    for thresh in thresholds:
        # Compute correct labels.
        print("\nComputing labels and splitting data ")
        if dtype == "atr":
            labelling_fn = compute_atr
        elif dtype == "fisher":
            labelling_fn = compute_fisher
        elif dtype == "back":
            labelling_fn = compute_background

        y_train = labelling_fn(labels_train, thresh)
        y_val = labelling_fn(labels_val, thresh)

        # Define sampler.
        class_weights = get_samples_weights(y_train)
        sampler = WeightedRandomSampler(class_weights.type("torch.DoubleTensor"), len(class_weights))
        # Define dataloaders.
        train_dataloader = DataLoader(DataModule(X_train, y_train), batch_size=1024, sampler=sampler)
        validation_dataloader = DataLoader(DataModule(X_val, y_val), batch_size=1024, shuffle=False)

        # Set device to avoid making mess with gpus.
        devices = get_freer_gpu()
        # MLflow experiment name. Must be unique.
        exp = "_".join([dtype, str(thresh), "optuna"])
        mlflow.set_experiment(exp)
        # Perform study.
        study = optuna.create_study(study_name=exp, direction="maximize", pruner=optuna.pruners.HyperbandPruner(), sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: objective(trial, exp, train_dataloader, validation_dataloader, devices), n_trials=50)

        best_trial = study.best_trial

        for key, value in best_trial.params.items():
            print(f"{key}: {value}")


if __name__=="__main__":
    Main()
