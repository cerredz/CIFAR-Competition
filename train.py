import torch
import torch.nn as nn
import torchvision
import os
import json
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as data
from torchvision import datasets
import torch.optim as optim
import torchvision.transforms as transforms
import sys
import numpy as np

class CifarNN(nn.Module):
    def __init__(self, input_features: int):
        super(CifarNN, self).__init__()
        self.input_features = input_features

        self.model = nn.Sequential(
            nn.BatchNorm1d(self.input_features),
            nn.Linear(self.input_features, 256),
            nn.ReLU(),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, features):
        x = torch.tensor(np.array(features))
        return self.model(x)

def train(batch_size: int, epochs: int, lr: float):
    # load datasets and initialize labels
    path = os.path.join(os.path.dirname(__file__), "data")
    dataset_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    datasets = {name: unpickle(os.path.join(path, name)) for name in dataset_names}
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # convert datasets int 1 and into pandas dataframe
    dataset = pd.DataFrame([
        {
            "data": data_item, 
            "labels": create_label_array(datasets[name][b"labels"][i])
        }
        for name in dataset_names 
        for i, data_item in enumerate(datasets[name][b"data"])
    ])

    X = (torch.tensor(np.array(dataset['data'].tolist()), dtype=torch.float32))
    y = torch.tensor(np.array(dataset['labels'].tolist()), dtype=torch.float32)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42, shuffle=True)
    
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # define model / loss
    input_features = 32 * 32 * 3
    model = CifarNN(input_features)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for bidx, (input_features, label) in enumerate(train_loader):
            optimizer.zero_grad() 
            pred = model(input_features)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            if (bidx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{bidx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    batch_size = 1024
    lr = 1e-4
    epochs=20
    train(batch_size,epochs, lr)
