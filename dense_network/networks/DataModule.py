import pandas as pd

import torch
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    """
    Dataset object plus methods for train test splid and kfold cross validation.
    """

    def __init__(
        self, samples: pd.DataFrame, targets: pd.Series
        ):
        """
        Initialization.
        :param samples: pd.DataFrame of descriptors.
        :parma series: pd.Series of ATR labels.
        """
        super().__init__()

        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1,1)


    def __len__(self):
        return len(self.targets)


    def __getitem__(self, idx:int):
        sample = self.samples[idx]
        target = self.targets[idx]
        return sample, target
