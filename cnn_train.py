import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from utils import *
from sklearn.model_selection import train_test_split
from augmentation import *

""" 
Implementation of a CNN for the Cifar-10 Dataset
"""
class CifarCnnDense(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Sequential(nn.Conv2d(3, 64, (3,3)), nn.BatchNorm2d(64), nn.GELU())
        self.l2 = nn.Sequential(nn.Conv2d(64, 128, (3,3)), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2)) 
        self.l3 = nn.Sequential(nn.Conv2d(128, 256, (3,3)), nn.BatchNorm2d(256), nn.GELU())
        self.l4 = nn.Sequential(nn.Conv2d(256, 256, (3,3)),  nn.BatchNorm2d(256), nn.GELU(), nn.MaxPool2d(2))
        self.l5 = nn.Sequential(nn.Conv2d(256, 512, (3,3)), nn.BatchNorm2d(512), nn.GELU())
        self.l6 = nn.Sequential(nn.Conv2d(512, 256, (3,3)), nn.BatchNorm2d(256), nn.GELU())
        self.flatten = nn.Flatten()
        self.dense1 = nn.Sequential(nn.Linear(256, 128), nn.Dropout(p=.5), nn.GELU())
        self.dense2 = nn.Sequential(nn.Linear(128, 64), nn.Dropout(p=.3), nn.GELU())
        self.dense3 = nn.Sequential(nn.Linear(64, 10), nn.Dropout(p=.2), nn.GELU())

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    
class CifarCnn(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.l1 = nn.Sequential(nn.Conv2d(3, 16, (3,3)), nn.BatchNorm2d(16), nn.GELU())
        self.l2 = nn.Sequential(nn.Conv2d(16, 48, (2,2)), nn.BatchNorm2d(48), nn.GELU())
        self.l3 = nn.Sequential(nn.Conv2d(48, 96, (2,2)), nn.BatchNorm2d(96), nn.GELU())
        self.l4 = nn.Sequential(nn.Conv2d(96, 200, (2,2)), nn.BatchNorm2d(200), nn.GELU(), nn.MaxPool2d(2))
        self.l5 = nn.Sequential(nn.Conv2d(200, 200, (2,2)), nn.BatchNorm2d(200), nn.GELU())
        self.l6 = nn.Sequential(nn.Conv2d(200, 400, (2,2)), nn.BatchNorm2d(400), nn.GELU())
        self.l7 = nn.Sequential(nn.Conv2d(400, 800, (2,2)), nn.BatchNorm2d(800), nn.GELU())
        self.l8 = nn.Sequential(nn.Conv2d(800, 1600, (2,2)), nn.BatchNorm2d(1600), nn.GELU())
        self.l9 = nn.Sequential(nn.Conv2d(1600, 10, (1,1)), nn.BatchNorm2d(10), nn.AdaptiveAvgPool2d((1,1)))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        return x

class CifarLitCnn(L.LightningModule):
    def __init__(self, cnn, lr, weight_decay):
        super().__init__()
        self.cnn = cnn
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        pred = self.cnn(x)
        loss = nn.CrossEntropyLoss()(pred, y)  # Expects logits and integer labels
        self.log("loss:", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.cnn(x)
        loss = nn.CrossEntropyLoss()(pred, y)
        preds = torch.argmax(pred, dim=1)
        accuracy = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.cnn(x)
        test_loss = nn.CrossEntropyLoss()(pred,y)
        preds = torch.argmax(pred, dim=1)
        accuracy = (preds == y).float().mean()
        self.log("test_loss", test_loss)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

def train(lr, batch_size, epochs, weight_decay):
    # Get training data
    path = os.path.join(os.path.dirname(__file__), "data")
    dataset_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    datasets = {name: unpickle(os.path.join(path, name)) for name in dataset_names}
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    # Get test data
    test_dataset_cifar = unpickle(os.path.join(path, "test_batch"))

    x = []
    y = []
    
    # Convert training / test data into 3 x 32 x 32 input for our 2d CNN
    for key, value in datasets.items():
        batch_data = value[b"data"]
        batch_labels = value[b"labels"]
        
        for i, (data, label) in enumerate(zip(batch_data, batch_labels)):
            data = convert_conv_2d_input(data)
            x.append(data)
            y.append(label)

    x_test = []
    y_test = []
    test_data = test_dataset_cifar[b"data"]
    test_labels = test_dataset_cifar[b"labels"]
    for i, (data, label) in enumerate(zip(test_data, test_labels)):
        data = convert_conv_2d_input(data)
        x_test.append(data)
        y_test.append(label)

    #  convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # split into train / val / test and then load into data loaders
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, train_size=.8, shuffle=True, random_state=42)

    augmented_data = []
    augmented_labels = []
    print("Augmenting data...")
    for data, label in zip(x_train, y_train):
        augmented_data.extend(noise_conv_2d(data))
        augmented_data.append(horizontal_flip(data))
        augmented_data.append(vertical_flip(data))
        augmented_data.append(scale(data))
        augmented_labels.extend([label] * 7) # noise x 4, hor flip, vert flip
    
    # Concatenate
    x_train = np.concatenate([x_train, augmented_data], axis=0)
    y_train = np.concatenate([y_train, augmented_labels], axis=0)
    
    # augment the training data
    train_loader = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_loader = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_loader = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_dataset = DataLoader(train_loader, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    val_dataset = DataLoader(val_loader, batch_size=batch_size, shuffle=False, num_workers=8,persistent_workers=True, pin_memory=True)
    test_dataset = DataLoader(test_loader, batch_size=batch_size, shuffle=False, num_workers=8,persistent_workers=True, pin_memory=True)

    # train the model
    model = CifarLitCnn(CifarCnnDense(), lr, weight_decay)
    #model = CifarLitCnn(CifarCnn(), lr, weight_decay)
    trainer = L.Trainer(max_epochs=epochs, log_every_n_steps=10)
    trainer.fit(model=model, train_dataloaders=train_dataset, val_dataloaders=val_dataset)
    trainer.test(model=model, dataloaders=test_dataset)

if __name__ == "__main__":
    lr = 1e-3
    batch_size = 2048
    epochs = 4
    weight_decay=1e-4
    train(lr, batch_size, epochs,weight_decay)