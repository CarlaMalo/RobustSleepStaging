import torch
import torch.nn as nn


class Conv1DBaseline(nn.Module):
    """Small 1D-CNN baseline for sleep staging on single-channel epochs.

    Input shape: (batch, 1, samples)
    Output: logits for 5 sleep stages (W,N1,N2,N3,REM)
    """
    def __init__(self, in_channels=1, n_classes=5, input_length=3000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        f = self.features(x)
        out = self.classifier(f)
        return out
