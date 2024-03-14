import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import os.path as osp
from tqdm import tqdm
import click


def _compute_p_values(bioactivity_matrix, assays):
    """
    Compute p-values derived from fisher exact test.
    """
    print("Running Fisher exact-test . . .")

    sample = bioactivity_matrix[assays]
    actives_sample = sample.sum(axis=1)
    
    other = bioactivity_matrix.drop(assays, axis=1)
    actives_other = other.sum(axis=1)

    # Compute pvalues with Fisher test - Which compounds have enriched activity for a certain wavelenght group?
    p_values = []
    for a, b in tqdm(zip(actives_sample, actives_other), total=len(bioactivity_matrix)):
        p_value = fisher_exact([[a, len(assays)], [b, len(bioactivity_matrix.colums) - len(assays)]], alternative='greater')[1]
        p_values.append(p_value)

    df = pd.DataFrame(bioactivity_matrix["smiles"].values, np.asarray(p_values)).T
    
    return df


def _compute_atr(bioactivity_matrix, assays):
    """
    Compute activity-to-tested ratio.
    """
    print("Computing ATR values . . .")

    atr_values = [] # store binarized atr values per technology
    # get assays ids for specific optical method.
    sample = bioactivity_matrix[assays] # assays with specific tech
    # other = bioactivity_matrix.drop(assays, axis=1) # other assays
    
    atr_sample = sample.sum(axis=1) / (len(sample.columns) - sample.isna().sum(axis=1))
    # atr_other = other.sum(axis=1) / (len(other.columns) - other.isna().sum(axis=1))

    # mean, std = atr_other.mean(), atr_other.std()

    atr_values.append(atr_sample)
    
    df = pd.DataFrame(bioactivity_matrix["smiles"].values, atr_values).T

    return df


def _compute_nar(bioactivity_matrix, background_matrix, assays):
    """
    Compute noise-to-active ratio.
    """
    print("Computing NAR values . . .")

    sample_main = bioactivity_matrix[assays]
    sample_back = background_matrix[assays]    
    
    sample = (sample_back == 1) &  (sample_main == 1)
    
    nar_values = sample.sum(axis=1) / (len(sample.columns) - sample.isna().sum(axis=1))
    
    df = pd.DataFrame(nar_values)

    return df


@click.command()
@click.option("--main", default="bioactivity_matrix.csv", help="csv file containing binary bioactivity matrix.")
@click.option("--back", default="background_matrix.csv", help="csv file containing binary background matrix.")
@click.option("--assays", default="fluorescent_assays_indexes.npy", help="array containing technology specific assay indexes.")
@click.option("--fisher", is_flag=True, default=True, help="compute Fisher test.")
@click.option("--atr", is_flag=True, default=True, help="compute ATR values.")
@click.option("--nar", is_flag=True, default=True, help="compute NAR values.")
def Main(main:str, back:str, assays:str, fisher:bool, atr:bool, nar:bool):
    """
    Run metrics computation
    """
    # Load data.
    bioactivity_matrix = pd.read_csv(osp.join("../data/raw", main), index_col=0)
    assays = np.load(osp.join("../data/raw", assays), allow_pickle=False)
    
    if fisher:
        p_values = _compute_p_values(bioactivity_matrix, assays)
        p_values.to_csv("../data/clean/p_values.csv")

    if atr:
        atr_values = _compute_atr(bioactivity_matrix, assays)
        atr_values.to_csv("../data/clean/atr_values.csv")

    if nar:
        background_matrix = [pd.read_csv(osp.join("../data/raw", back), index_col=0)]
        nar_values = _compute_nar(bioactivity_matrix, background_matrix, assays)
        nar_values.to_csv("../data/clean/nar_values.csv")

    return (f"Metrics computed: fisher:{fisher}, atr:{atr}, nar:{nar}")

if __name__ == "__main__":
    Main()
