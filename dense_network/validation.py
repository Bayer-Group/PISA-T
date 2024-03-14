import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import click
import os
import os.path as osp

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.metrics import recall_score, precision_score, matthews_corrcoef, roc_auc_score, average_precision_score

from networks.SimpleClassifier import SimpleClassifier
from networks.DataModule import DataModule
from networks.engine import get_freer_gpu

from compute_labels import compute_atr, compute_fisher, compute_background

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

"""
Test optimized neural network: propietary test set.
"""
# Data directory.
datadir = "../data/raw"


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


def classification_report(true, preds):
    """
    Make classification report.
    """
    classification_metrics = {"recall":0, "precision":0, "mcc":0, "auroc":0, "aupr":0}
    
    if len(np.unique(true))==1:
        return classification_metrics

    binary = [1 if i >= 0.5 else 0 for i in preds]
    
    classification_metrics["recall"] = recall_score(true, binary)
    classification_metrics["precision"] = precision_score(true, binary)
    classification_metrics["mcc"] = matthews_corrcoef(true, binary)
    classification_metrics["auroc"] = roc_auc_score(true, preds)
    classification_metrics["aupr"] = average_precision_score(true, preds)
    
    return classification_metrics


@click.command()
@click.option("-d", "--dtype", required=True, help="metric to be used: atr/nar/fisher.")
def Main(dtype):
    """
    Test DNN.
    """
    # Load data.
    print("\n Loading data ...")

    smiles = pd.read_csv(osp.join(datadir, "bioactivity_matrix.csv")).smiles.values
    fingerprints = np.array([_compute_morgan(i) for i in tqdm(smiles)], dtype=float)

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

    # Load the optimized network parameters.
    params = pd.read_csv(f"results/validation/params_{dtype}.csv", index_col=0)
    # Test at different thresholds.
    predictions_dict = {str(i): None for i in thresholds}
    for thresh in thresholds:

        # Set device to avoid making mess with gpus.
        devices = get_freer_gpu()
        # Select right parameters.
        optimized_params= params[str(thresh)].to_dict()
        # Instanciate model.
        model = SimpleClassifier(input_size=2048, params=optimized_params)
        # Initialize trainer.
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="gpu",
            devices=[devices],
            logger=False,
            callbacks=[
                EarlyStopping(monitor="val_mcc", mode="max")
                ]
        )

        print(f"\nInitilized Model: {model}")

        # Compute correct labels.
        print("\tComputing labels and splitting data ")
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
        val_dataloader = DataLoader(DataModule(X_val, y_val), batch_size=1024, shuffle=False)

        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # Make predictions.
        with torch.no_grad():
            predictions = torch.sigmoid(model(torch.tensor(X_val, dtype=torch.float32)))
        predictions = predictions.numpy()

        results = classification_report(y_val, predictions)
        print(results)

        predictions_dict[str(thresh)] = predictions.ravel()

    print(predictions_dict)
    
    # Save predictions.
    predictions_df = pd.DataFrame(predictions_dict)
    predictions_df.to_csv(f"results/validation/{dtype}.csv")
    
    print("Results saved") 


if __name__=="__main__":
    Main()
