import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import json
from utils import unpickle


class CifarNN(nn.Module):
    def __init__(self, input_features: int):
        super(CifarNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(self.input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 66),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Softmax(),
        )

    def forward(self):
        pass

# Load CIFAR-10 dataset
def train():
    path = os.path.join(os.path.dirname(__file__), "data")
    dataset_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    datasets = {name: unpickle(os.path.join(path, name)) for name in dataset_names}
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print(datasets["data_batch_1"][b"data"])
    
if __name__ == "__main__":
    train()
