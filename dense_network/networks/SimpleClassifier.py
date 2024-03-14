import numpy as np
import os

import torch
from torch import nn

import pytorch_lightning as pl
from torchmetrics import MatthewsCorrCoef


class SimpleClassifier(pl.LightningModule):
    """
    Binary Classifier for fluorescence interference prediction.
    """

    def __init__(self, input_size:int, params:dict):
        """
        :param input_size: integer defining the size of the input vector
        :param params: network architecture parameters dict(num_layers: int, num_units: int, 
                                                        dropout: float, lr: float
                                                    )
        """
        super().__init__()

        in_features = input_size

        self.params = params
        # Define loss function.
        self.loss_fn = nn.BCEWithLogitsLoss()
        # Define classification metric.
        self.matthews = MatthewsCorrCoef(task="binary", num_classes=2, threshold=0.5)

        layers = []
        for val in range(int(self.params["num_layers"])):
            out_features = int(self.params["num_units"]) // (val+1)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.params["dropout"]))

            in_features = out_features

        layers.append(nn.Linear(out_features, 1))

        self.classifier = nn.Sequential(*layers)


    def forward(self, x):
        """
        Output predictions.
        """
        logits = self.classifier(x)
        return logits


    def training_step(self, batch, batch_idx):
        """
        Complete training loop for one training batch.
        """
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)

        binary = torch.sigmoid(pred)
        corrcoeff =  self.matthews(binary, y.int())

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_mcc", corrcoeff, on_epoch=True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        """
        Validation loop for one test batch.
        """
        x, y = batch
        pred = self.classifier(x)
        loss = self.loss_fn(pred, y)

        binary = torch.sigmoid(pred)
        corrcoeff =  self.matthews(binary, y.int())

        self.log("val_loss", loss)
        self.log("val_mcc", corrcoeff)

        return loss


    def test_step(self, batch, batch_idx):
        """
        Validation loop for one test batch.
        """
        x, y = batch
        pred = self.classifier(x)
        loss = self.loss_fn(pred, y)

        binary = torch.sigmoid(pred)
        corrcoeff =  self.matthews(binary, y.int())

        self.log("test_loss", loss)
        self.log("test_mcc", corrcoeff)

        return loss


    def configure_optimizers(self):
        """
        Configure optimizer for backprop.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params["lr"])
        return optimizer
