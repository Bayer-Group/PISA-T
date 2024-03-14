import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import click
import os.path as osp

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import recall_score, precision_score, matthews_corrcoef, roc_auc_score, average_precision_score

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from compute_labels import compute_atr, compute_fisher, compute_background

"""
Baseline Random Forest Classifier: Validation with hyperparameters search.
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
    Get train and test splits.
    """
    # Get train and validation splits.
    train_index = np.load("../data/splits/train_indices.npy", allow_pickle=False)
    test_index = np.load("../data/splits/val_indices.npy", allow_pickle=False)

    X_train, y_train = fingerprints[train_index], labels.loc[train_index]
    X_test, y_test = fingerprints[test_index], labels.loc[test_index]

    return X_train, y_train, X_test, y_test


def make_rfc(params):
    """
    Instanciate Random Forest with specified parameters.
    """
    rfc = BalancedRandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            bootstrap=params["bootstrap"],
            n_jobs=16,
            verbose=1,
        )

    return rfc


def classification_report(true, preds):
    """
    Make classification report.
    """
    classification_metrics = {"recall":0, "precision":0, "mcc":0, "auroc":0, "aupr":0}
    
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
    Baseline Random Forest Classifier.
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
    X_train, labels_train, X_test, labels_test = train_val_split(fingerprints, interference_metric)

    # Load the optimized random forest parameters.
    params = pd.read_csv(f"results/validation/params_{dtype}.csv", index_col=0)

    # Test at different thresholds.
    predictions_dict = {str(i): None for i in thresholds}
    for thresh in thresholds:
        
        # select right parameters.
        params_at_threshold = {
            "bootstrap": np.ceil(params[str(thresh)]["bootstrap"]).astype(bool),
            "max_depth": np.ceil(params[str(thresh)]["max_depth"]).astype(int),
            "n_estimators": np.ceil(params[str(thresh)]["n_estimators"]).astype(int)
        }

        # Compute correct labels.
        print("\tComputing labels and splitting data ")
        if dtype == "atr":
            labelling_fn = compute_atr
        elif dtype == "fisher":
            labelling_fn = compute_fisher
        elif dtype == "back":
            labelling_fn = compute_background

        y_train = labelling_fn(labels_train, thresh)
        y_test = labelling_fn(labels_test, thresh)

        # Define the model.
        model = make_rfc(params_at_threshold)

        print(f"\nInitilized Model: {model}")

        # Train and validate
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_test)[:, 1]

        results = classification_report(y_test, predictions)
        print(results)

        predictions_dict[str(thresh)] = predictions

    # Save predictions.
    predictions_df = pd.DataFrame(predictions_dict)
    predictions_df.to_csv(f"results/validation/{dtype}.csv")

    print("Results saved")


if __name__=="__main__":
    Main()
