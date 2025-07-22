import torch
import torch.nn as nn
import torchvision
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from augmentation import *

class CifarNN(pl.LightningModule):
    def __init__(self, input_features: int, lr: float = 1e-5, weight_decay: float = 0.01):
        super(CifarNN, self).__init__()
        self.input_features = input_features
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        self.model = nn.Sequential(
            nn.BatchNorm1d(self.input_features),
            nn.Linear(self.input_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.45),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 10), 
        )

    def forward(self, features):
        return self.model(features)

    def training_step(self, batch, batch_idx):
        input_features, labels = batch
        input_features = input_features.view(-1, self.input_features)
        pred = self(input_features)  # Logits
        loss = nn.CrossEntropyLoss()(pred, labels)  # Expects logits and integer labels
        self.log('train_loss', loss, prog_bar=True)
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        input_features, labels = batch
        input_features = input_features.view(-1, self.input_features)
        pred = self(input_features)  # Logits
        loss = nn.CrossEntropyLoss()(pred, labels)
        probs = torch.softmax(pred, dim=1)  # Explicitly compute probabilities
        preds = torch.argmax(probs, dim=1)  # Predicted classes
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        output = {'val_loss': loss, 'preds': preds, 'labels': labels, 'probs': probs}
        self.validation_step_outputs.append(output)
        self.val_losses.append(loss.item())
        return output
    
    def test_step(self, batch, batch_idx):
        input_features, labels = batch
        input_features = input_features.view(-1, self.input_features)
        pred = self(input_features)  # Logits
        loss = nn.CrossEntropyLoss()(pred, labels)
        probs = torch.softmax(pred, dim=1)  # Explicitly compute probabilities
        preds = torch.argmax(probs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        output = {'test_loss': loss, 'preds': preds, 'labels': labels, 'probs': probs}
        self.test_step_outputs.append(output)
        self.test_losses.append(loss.item())
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        cm = confusion_matrix(labels.cpu(), preds.cpu(), labels=range(10))
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        self.logger.experiment.add_figure('confusion_matrix', fig, self.current_epoch)
        precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average=None)
        for i in range(10):
            self.log(f'precision_class_{i}', precision[i], prog_bar=False)
            self.log(f'recall_class_{i}', recall[i], prog_bar=False)
            self.log(f'f1_class_{i}', f1[i], prog_bar=False)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        avg_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('test_loss_epoch', avg_test_loss, prog_bar=True)
        self.test_losses.append(avg_test_loss.item())
        self.test_step_outputs.clear()

    def on_train_end(self):
        train_loss_avg = np.mean(self.train_losses)
        val_loss_avg = np.mean(self.val_losses)
        test_loss_avg = np.mean(self.test_losses)
        
        bias = max(0, train_loss_avg - 0.1)
        variance = max(0, val_loss_avg - test_loss_avg)
        noise = max(0, test_loss_avg - (train_loss_avg + variance))

        fig, ax = plt.subplots()
        components = ['Bias', 'Variance', 'Noise']
        values = [bias, variance, noise]
        ax.bar(components, values, color=['#FF6384', '#36A2EB', '#FFCE56'])
        ax.set_ylabel('Error Value')
        ax.set_title('Bias-Variance-Noise Decomposition')
        ax.set_ylim(bottom=0)
        
        self.logger.experiment.add_figure('bias_variance_noise', fig, global_step=0)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

def train(batch_size: int, epochs: int, lr: float, weight_decay: float):
    path = os.path.join(os.path.dirname(__file__), "data")
    dataset_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    datasets = {name: unpickle(os.path.join(path, name)) for name in dataset_names}
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    # Collect all data and labels properly
    # First, split the original data
    original_data = []
    original_labels = []

    for name in dataset_names:
        batch_data = datasets[name][b"data"]
        batch_labels = datasets[name][b"labels"]
        original_data.extend(batch_data)
        original_labels.extend(batch_labels)

    X_orig = torch.tensor(np.array(original_data), dtype=torch.float32)
    y_orig = torch.tensor(original_labels, dtype=torch.long)

    # Split ORIGINAL data first
    x_train_orig, x_val, y_train_orig, y_val = train_test_split(
        X_orig, y_orig, test_size=0.3, random_state=42, shuffle=True
    )

    # THEN apply augmentation only to training data
    augmented_train_data = []
    augmented_train_labels = []

    for i, image in enumerate(x_train_orig):
        augmented_images = augment([image.numpy()])
        for aug_image in augmented_images:
            augmented_train_data.append(aug_image)
            augmented_train_labels.append(y_train_orig[i])

    x_train = torch.tensor(np.array(augmented_train_data), dtype=torch.float32)
    y_train = torch.tensor(augmented_train_labels, dtype=torch.long)
    
    test_dataset = unpickle(os.path.join(path, "test_batch"))
    x_test = torch.tensor(test_dataset[b"data"], dtype=torch.float32)
    y_test = torch.tensor(test_dataset[b"labels"], dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_features = 32 * 32 * 3
    model = CifarNN(input_features, lr=lr, weight_decay=weight_decay)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger("./runs", name=f"train_{timestamp}")
    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    images, labels = next(iter(train_loader))
    images = images.view(-1, 3, 32, 32)
    grid = torchvision.utils.make_grid(images[:8])
    logger.experiment.add_image("sample_images", grid, 0)
    simple_model = nn.Sequential(*model.model)
    logger.experiment.add_graph(simple_model, images.view(-1, input_features))
    sample_size = min(1000, len(x_train))
    sample_indices = torch.randperm(len(x_train))[:sample_size]
    embedding_data = x_train[sample_indices]
    embedding_labels = y_train[sample_indices]
    if embedding_data.dim() > 2:
        embedding_data = embedding_data.view(embedding_data.size(0), -1)
    logger.experiment.add_embedding(
        embedding_data,
        metadata=embedding_labels.tolist(),
        global_step=0,
        tag="cifar_embeddings"
    )

    model.eval()
    with torch.no_grad():
        sample_images, sample_labels = next(iter(val_loader))
        sample_images = sample_images.view(-1, 3, 32, 32)
        preds = model(sample_images.view(-1, input_features))
        probs = torch.softmax(preds, dim=1)  # Explicit probabilities
        pred_labels = torch.argmax(probs, dim=1)
        true_labels = sample_labels
        grid = torchvision.utils.make_grid(sample_images[:8])
        logger.experiment.add_image("sample_predictions", grid, 0)
        pred_text = "\n".join([f"Img {i}: Pred={labels[pred_labels[i]]}, True={labels[true_labels[i]]}, Probs={probs[i].cpu().numpy().round(3)}" for i in range(8)])
        logger.experiment.add_text("sample_pred_labels", pred_text, 0)
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.view(-1, input_features)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    val_accuracy = correct / total
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    return val_accuracy

if __name__ == "__main__":
    batch_size = 1024
    lr = 1e-5
    epochs = 50
    weight_decay = 0.5
    hyperparameters = [
        {"batch_size": 1024, "epochs": 25, "lr": 1e-4, "weight_decay": 0.1},
        
    ]
    max_val_accuracy = 0
    best_parameters = None
    for parameters in hyperparameters:
        val_acc = train(parameters["batch_size"], parameters["epochs"], parameters["lr"], parameters["weight_decay"])
        if val_acc > max_val_accuracy:
            best_parameters = parameters
            max_val_accuracy = val_acc
    print("best hyperparameters ", best_parameters)
    print("best accuracy", max_val_accuracy)

