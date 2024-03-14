import pandas as pd
import numpy as np


def compute_fisher(p_values, thresh):
    """
    Compute interference labels based on fisher test performed on main signal.
    """
    labels = (p_values.values <= thresh).astype(int)

    return labels


def compute_atr(atr_values, thresh):
    """
    Compute interference labels based on ATR computed on main signal.
    """
    atr_samples= atr_values
    
    labels = (atr_samples >= atr_samples.mean() + thresh*atr_samples.std()).astype(int)

    return labels


def compute_background(nar_values, thresh):
    """
    Compute interference labels based on background/main signals comparison.
    """
    # Compute binary labels.
    labels = (nar_values >= thresh).astype(int)

    return labels

