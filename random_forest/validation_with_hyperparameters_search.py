import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import click
import os.path as osp

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import recall_score, precision_score, matthews_corrcoef, roc_auc_score, average_precision_score

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from bayes_opt import BayesianOptimization

from compute_labels import compute_atr, compute_fisher, compute_background

import warnings
warnings.filterwarnings("ignore")

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


def save_results(validation_parameters, validation_metrics, datatype):
    """
    Save results.
    """
    params_df = pd.DataFrame.from_dict(validation_parameters)
    metrics_df = pd.DataFrame(validation_metrics, index=["mcc"])
    params_df.to_csv(f"results/validation/params_{datatype}.csv")
    metrics_df.to_csv(f"results/validation/metrics_{datatype}.csv")
    
    return print("Results saved.")


@click.command()
@click.option("-d", "--dtype", required=True, help="fisher/atr/back")
def Main(dtype):
    """
    Hyperaparameters search for Random Forest Classifier.
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

    # Validate at different thresholds.
    validation_parameters = {i: None for i in thresholds}
    validation_metrics = {i: None for i in thresholds}
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

        val = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val
        }

        print(f"\n Running Bayesian optimization for {dtype}")

        # Optimize
        def black_box_function(n_estimators, max_depth, bootstrap):
            """
            Performance optimization.
            """
            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'bootstrap': bootstrap
            }
            classification_metrics = run_validation(val, params)
            return classification_metrics['mcc']

        def function_to_be_optimized(n_estimators, max_depth, bootstrap):
            """
            Parameters optimization with bayesian optimization.
            """
            est = int(n_estimators)
            max_d = int(max_depth)
            boot = bool(int(bootstrap))
            return black_box_function(
                            n_estimators=est, 
                            max_depth=max_d, 
                            bootstrap=boot,
                            )

        def run_validation(val, params):
            """
            Fit training data and compute classification metrics on the validation.
            """
            model = BalancedRandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        bootstrap=params["bootstrap"],
                        n_jobs=32,
                    )
            model.fit(val["X_train"], val["y_train"])
            predictions = model.predict_proba(val["X_val"])[:,1]

            results = classification_report(val["y_val"], predictions)

            return results


        def run_bayesian_optimization(val):
            # Run the random search in cross validation for all the datasets and save the parameters giving the best MCC.
            bounds = {
            'n_estimators': (200, 2000), 
            'max_depth': (2, 101), 
            'bootstrap': (0, 1.9),
            }


            optimizer = BayesianOptimization(
            f=function_to_be_optimized,
            pbounds=bounds,
            verbose=2,
            random_state=1,
            )

            optimizer.set_gp_params(alpha=1e-3)
            optimizer.maximize(
            init_points=2,
            n_iter=50,
            )
            params_val = optimizer.max['params']
            results_val = optimizer.max['target']

            return params_val, results_val

        results = run_bayesian_optimization(val)
        
        validation_parameters[thresh] = results[0]
        validation_metrics[thresh] = results[1]
        
        print(results)

    # Save best results.
    save_results(validation_parameters, validation_metrics, dtype)


if __name__=="__main__":
    Main()
