import torch
import torch.nn as nn
import torch.nn.functional as F

class LogCNN(nn.Module):
    def __init__(self, num_classes):
        super(LogCNN, self).__init__()

        # Feature extractor: Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),  # First convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2),  # Max pooling layer
            nn.Conv2d(16, 32, kernel_size=2),  # Second convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2)  # Max pooling layer
        )

        # Define the classifier, which will be initialized dynamically after convolution layers
        self.classifier = None
        self.num_classes = num_classes

    def forward(self, x):
        # Ensure the input shape is [batch_size, 1, height, width]
        if x.dim() == 3:  # If 3D input (e.g., [batch_size, 1, length])
            x = x.unsqueeze(2)  # Add a dimension for height (e.g., [batch_size, 1, 1, length])

        # Pass through the convolutional layers
        x = self.features(x)

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)

        # Dynamically create the classifier layers based on the flattened size
        flattened_size = x.size(1)
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.num_classes)
        )

        # Forward pass through the classifier
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
