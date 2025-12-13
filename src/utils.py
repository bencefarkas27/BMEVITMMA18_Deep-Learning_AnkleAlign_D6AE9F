import logging
import sys
import os
import numpy as np
from networks import BaseLineNetWork, BestNetWork
import torch
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

def setup_logger(name=__name__):
    """
    Sets up a logger that outputs to the console (stdout).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def check_cuda(logger):
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"  Compute Capability: {props.major}.{props.minor}")
        device = "cuda" 
    else:
        logger.info("CUDA not available")
        device = "cpu"
    return device

def create_torch_dataloader(logger, data, preped_folder, transform, batch_size=16, shuffle=False, val_split=None):
    """
    Creates a PyTorch DataLoader from image data and labels. If val_split is provided, splits data into training and validation sets.
    """
    images = []
    labels = []

    for img_name, label in data:
        img_path = os.path.join(preped_folder, img_name)
        try:
            img = Image.open(img_path).convert('L')
            img_tensor = transform(img)
            images.append(img_tensor)
            labels.append(label)
        except Exception as e:
            logger.info(f"Error loading {img_name}: {e}")

    images_tensor = torch.stack(images)


    label_to_idx = {label: idx for idx, label in enumerate(np.unique(labels))}
    logger.info(f"Label mapping: {label_to_idx}")
    labels_encoded = [label_to_idx[label] for label in labels]
    labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)

    if val_split is not None:
        images_tensor, images_val_tensor, labels_tensor, labels_val_tensor = train_test_split(
            images_tensor, labels_tensor, test_size=val_split, random_state=42, stratify=labels_tensor)
        train_dataset = TensorDataset(images_tensor, labels_tensor)
        val_dataset = TensorDataset(images_val_tensor, labels_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        dataset = TensorDataset(images_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader
    
def get_transforms():
    """
    Returns image transformations for training or inference.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def load_best_model(model_path, device):
    model = BestNetWork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_baseline_model(model_path, device):
    model = BaseLineNetWork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
