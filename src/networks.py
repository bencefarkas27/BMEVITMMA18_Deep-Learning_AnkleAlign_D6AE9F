import torch
import torch.nn as nn

class BaseLineNetWork(torch.nn.Module):
    def __init__(self):
        super(BaseLineNetWork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 224x224 -> 112x112
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 112x112 -> 56x56
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 56x56 -> 28x28
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class BestNetWork(torch.nn.Module):
    def __init__(self):
        super(BestNetWork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),      # (3x3x1)x8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # 224x224 -> 112x112

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),       # (3x3x8)x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # 112x112 -> 56x56

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),       # (3x3x16)x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # 56x56 -> 28x28 

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),       # (3x3x32)x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # 28x28 -> 14x14 

            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x