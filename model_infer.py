import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from logcnn import LogCNN

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset class
class LogDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# Padding and reshaping
def pad_and_reshape(data):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    padded = F.pad(data_tensor.unsqueeze(1), pad=(1, 1, 1, 1))  # (N, 1, D+2, 1+2)
    return padded

# Corrected model loading from 'model.pt'
def load_saved_model(model_path='model.pt', input_length=256):
    model = LogCNN(input_length=input_length).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# Inference
def infer_anomalies(model, test_data, test_lines, batch_size=128):
    input_tensor = pad_and_reshape(test_data)
    test_loader = DataLoader(LogDataset(input_tensor), batch_size=batch_size)

    anomaly_lines = []
    idx = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            _, prob = model(batch)
            preds = torch.argmax(prob, dim=1).cpu().numpy()

            for pred in preds:
                if pred == 1:
                    anomaly_lines.append(test_lines[idx])
                idx += 1

    return anomaly_lines
