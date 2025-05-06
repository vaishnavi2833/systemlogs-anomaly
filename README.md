# Log Anomaly Detection with LogCNN

This repository contains scripts for detecting anomalies in system log files using a pre-trained LogCNN model.

## Overview

The system uses a Convolutional Neural Network (CNN) to identify anomalous patterns in log files. The process involves:

1. Parsing raw log files into a structured format
2. Processing the structured logs into feature vectors
3. Using the pre-trained CNN model to detect anomalies
4. Outputting the results

## Scripts

### `detect_anomalies.py`

This is the main script that handles the complete pipeline from raw log files to anomaly detection results.

#### Usage

```bash
python detect_anomalies.py --input <path_to_log_file> [--output <output_file>] [--model <model_path>]
```

Arguments:
- `--input`: Path to the input log file (required)
- `--output`: Path to save the anomaly detection results (default: anomalies.txt)
- `--model`: Path to the pre-trained model (default: log_cnn_model.pkl)

Example:
```bash
python detect_anomalies.py --input uploaded_logs/HDFS.log --output anomalies.txt
```

### `predict_anomalies.py`

This is a simplified script that assumes the input data has already been processed into the required format. It's useful for quick testing or integration into other systems.

#### Usage

```bash
python predict_anomalies.py --input <path_to_processed_data> [--output <output_file>] [--model <model_path>]
```

Arguments:
- `--input`: Path to the input numpy array file (.npy) containing processed log data (required)
- `--output`: Path to save the anomaly indices (default: anomaly_indices.txt)
- `--model`: Path to the pre-trained model (default: log_cnn_model.pkl)

Example:
```bash
python predict_anomalies.py --input processed_data/x_test_tf-idf_v5.npy --output anomaly_indices.txt
```

## Directory Structure

- `uploaded_logs/`: Directory for storing raw log files
- `parsed_logs/`: Directory for storing parsed log files
- `processed_data/`: Directory for storing processed data ready for model input
- `log_cnn_model.pkl`: Pre-trained LogCNN model

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- pandas
- scikit-learn

## Model Details

The LogCNN model is a convolutional neural network designed to detect anomalies in log data. It processes log sequences as 2D images and uses convolutional layers to identify patterns indicative of anomalies.

The model architecture consists of:
- Convolutional layers for feature extraction
- Max pooling layers for dimensionality reduction
- Fully connected layers for classification

The model outputs a binary classification: 0 for normal logs and 1 for anomalous logs.
