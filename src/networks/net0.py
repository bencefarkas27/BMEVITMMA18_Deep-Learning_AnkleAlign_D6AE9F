import torch

net0 = torch.nn.Sequential(
    torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 224x224 -> 112x112
    torch.nn.ReLU(),
    torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 112x112 -> 56x56
    torch.nn.ReLU(),
    torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 56x56 -> 28x28
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(32, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 3)                       # Output layer     
)