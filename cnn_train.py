import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as L
from augmentation import *

""" 
Implementation of a CNN for the Cifar-10 Dataset
"""
class CifarCnnDense(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Sequential(nn.Conv2d(3, 32, (3,3), stride=2), nn.ReLU())
        self.l2 = nn.Sequential(nn.Conv2d(32, 96, (5,5), stride=2), nn.ReLU())
        self.l3 = nn.Sequential(nn.Conv2d(96, 32, (3,3), stride=1), nn.ReLU())
        self.l4 = nn.Sequential(nn.Conv2d(32, 16, (3,3), stride=1), nn.ReLU())  # Adjusted kernel size
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(4 * 4 * 16, 128)  # Input: 256, Output: 128
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(128, 64)  # Output layer for classification
        self.dense3 = nn.Linear(64, 32)  # Output layer for classification
        self.dense4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.relu(x)
        x = self.dense4(x)
        return x
    
class CifarCnn(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


class CifarLitCnn(L.LightningModule):
    def __init__(self, cnn, lr):
        super().__init__()
        self.cnn = cnn
        self.lr = lr

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.cnn(x)
        loss = nn.CrossEntropyLoss()(pred, y)  # Expects logits and integer labels
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def train():
    path = os.path.join(os.path.dirname(__file__), "data")
    dataset_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    datasets = {name: unpickle(os.path.join(path, name)) for name in dataset_names}
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


if __name__ == "__main__":
    train()