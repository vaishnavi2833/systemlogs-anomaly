# train_logcnn.py

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from logcnn import LogCNN  
import pickle

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Constants
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10
NUM_CLASSES = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset
class LogDataset(Dataset):
    def __init__(self, data, labels=None):
        self.X = data
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.long)
            return x, y
        return x

# Padding function
def pad_and_reshape(x):
    x = torch.tensor(x, dtype=torch.float32)
    x = F.pad(x, pad=(1, 1, 1, 1))  # Pad height and width
    return x.unsqueeze(1).numpy()  # Add channel dimension

# Compute F1
def compute_f1(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, prob = model(x)
            y_hat = torch.argmax(prob, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_hat.cpu().numpy())
    return f1_score(y_true, y_pred, average='macro')

# Training loop
def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_f1 = compute_f1(model, train_loader)
        val_f1 = compute_f1(model, val_loader)
        print(f"Epoch {epoch+1} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

    return model


# Anomaly prediction
def predict_and_find_anomalies(model, test_loader, original_test_lines):
    model.eval()
    anomaly_lines = []
    idx = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(DEVICE)
            _, prob = model(x)
            preds = torch.argmax(prob, dim=1)
            for pred in preds.cpu().numpy():
                if pred == 1:
                    anomaly_lines.append(original_test_lines[idx])
                idx += 1
    return anomaly_lines

# Save model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Main pipeline
def run_pipeline(train_data_path, train_labels_path, test_data_path, test_labels_path, original_test_lines_path, output_path):
    with open(original_test_lines_path, 'r') as f:
        original_test_lines_list = [line.strip() for line in f.readlines()]

    train_data = np.load(train_data_path)
    train_labels = (pd.read_csv(train_labels_path)['Label'] == 'Anomaly').astype(int)
    test_data = np.load(test_data_path)
    test_labels = (pd.read_csv(test_labels_path)['Label'] == 'Anomaly').astype(int)

    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=RANDOM_SEED)
    train_labels = train_labels.reset_index(drop=True)
    val_labels = val_labels.reset_index(drop=True)

    model = LogCNN(num_classes=NUM_CLASSES).to(DEVICE)

    train_data = pad_and_reshape(train_data)
    val_data = pad_and_reshape(val_data)
    test_data = pad_and_reshape(test_data)

    train_loader = DataLoader(LogDataset(train_data, train_labels), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(LogDataset(val_data, val_labels), batch_size=BATCH_SIZE)
    test_loader = DataLoader(LogDataset(test_data, test_labels), batch_size=BATCH_SIZE)

    model = train_model(model, train_loader, val_loader)
    save_model(model, "log_cnn_model.pt")

    anomalies = predict_and_find_anomalies(model, test_loader, original_test_lines_list)
    with open(output_path, 'w') as f:
        for line in anomalies:
            f.write(line + '\n')

    print(f"Anomalous lines saved to {output_path}")

    # Evaluation
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, prob = model(x)
            y_hat = torch.argmax(prob, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_hat.cpu().numpy())

    print("\nTest Set Metrics:")
    print("F1 Score:", f1_score(y_true, y_pred, average='macro'))
    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Optional call
if __name__ == "__main__":
    run_pipeline(
        train_data_path='./processed_data/x_train_tf-idf_v5.npy',
        train_labels_path='./processed_data/y_train_tf-idf_v5.csv',
        test_data_path='./processed_data/x_test_tf-idf_v5.npy',
        test_labels_path='./processed_data/y_test_tf-idf_v5.csv',
        original_test_lines_path='./uploaded_logs/HDFS.log',
        output_path='anomalous_lines.txt'
    )
