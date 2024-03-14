import pandas as pd
import numpy as np
import random
import os.path as osp
from tqdm import tqdm
import click
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _compute_scaffolds(smile):
    """
    Compute the Murcko scaffold for the SMILES.
    """
    mol = Chem.MolFromSmiles(smile)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold


@click.command()
@click.option("--smi", default="bioactivity_matrix", help="SMILES list or series.")
def Main(smi:str):
    """
    Run scaffold split.
    """
    # Load data.
    smiles = pd.read_csv(osp.join("../data/raw", smi))["smiles"]

    # Compute scaffold and save smile index to corresponding scaffold SMILES in the dictionary.
    scaffolds_dict = {}
    for idx, smile in tqdm(enumerate(smiles), total=len(smiles)):
        scaffold = _compute_scaffolds(smile) # compute scaffold
        scaffolds_dict[scaffold] = scaffolds_dict.get(scaffold, [])
        scaffolds_dict[scaffold].append(idx) # append smile index to corresponding scaffold

    # Sanity check: control if all compounds have been assigned to a scaffold.
    compounds = [len(i) for i in scaffolds_dict.values()]
    print(sum(compounds) == len(smiles))

    # Sort scaffolds dictionary keys from largest to smallest.
    scaffold_sorted = np.array(sorted(scaffolds_dict, key=lambda k: len(scaffolds_dict[k]), reverse=True))

    # Compute fingerprints.
    fingerprints = []
    for i in tqdm(scaffold_sorted):
        try:
            fps = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i),2,nBits=1024)
        except:
            if i == "O=c1[c-]cnn1-c1ccccc1":
                i = "O=c1cnnc1-c1ccccc1" # manually kekulize the molecule
                fps = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i),2,nBits=1024)
        
        fingerprints.append(fps)

    scaffolds_aggregated = {}
    
    repeated = set()
    for idx, key in tqdm(enumerate(scaffold_sorted), total=len(scaffold_sorted)):
        if key not in repeated:
            scaffolds_aggregated[key] = scaffolds_aggregated.get(key, scaffolds_dict[key]) # get scaffold smile and associated smiles indexes.
            repeated.update([key]) # add the scaffold to the repeated set.
            
            similarity = np.array(DataStructs.BulkTanimotoSimilarity(fingerprints[idx], fingerprints[idx+1:]))
            similarity = np.where(similarity >= 0.7)[0]
            smiles = scaffold_sorted[idx+1:][similarity]
            
            for i in smiles:
                if i not in repeated:
                    scaffolds_aggregated[key] += scaffolds_dict[i]
            
            repeated.update(smiles)
        
        else:
            continue

    # Check that all the compounds have been considered.
    count = 0
    for i in scaffolds_aggregated.values():
        count += len(i)

    # Define size of the splits.
    train_size = int(0.8 * len(scaffolds_aggregated))
    val_size = int(0.1 * len(scaffolds_aggregated))
    test_size = len(scaffolds_aggregated) - train_size - val_size

    scaffolds = list(scaffolds_aggregated.keys())

    # Get train indices.
    train_scaffolds = random.sample(scaffolds, k=train_size)

    train_indices = []
    for i in train_scaffolds:
        train_indices += scaffolds_aggregated[i]

    # Get validation indices.
    scaffolds = list(set(scaffolds) - set(train_scaffolds)) # remove already selected scaffolds
    val_scaffolds = random.sample(scaffolds, k=val_size)

    val_indices = []
    for i in val_scaffolds:
        val_indices += scaffolds_aggregated[i]

    # Get test indices.
    scaffolds = list(set(scaffolds) - set(val_scaffolds)) # remove already selected scaffolds
    test_scaffolds = random.sample(scaffolds, k=test_size)

    test_indices = []
    for i in test_scaffolds:
        test_indices += scaffolds_aggregated[i]

    # Sanity check: check that no indexes are repeated through the splits.
    len(set(train_indices + val_indices + test_indices))

    np.save(osp.join("../data/splits", "train_split"), np.array(train_indices), allow_pickle=False)
    np.save(osp.join("../data/splits", "val_split"), np.array(val_indices), allow_pickle=False)
    np.save(osp.join("../data/splits"," test_split"), np.array(test_indices), allow_pickle=False)

    return ("Dataset splitted by scaffold.")


if __name__ == "__main__":
    Main()
