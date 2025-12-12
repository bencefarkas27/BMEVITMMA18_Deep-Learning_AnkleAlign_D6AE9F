import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from tqdm.auto import tqdm

import logging
import sys

from utils import setup_logger
import config

import net0

logger = setup_logger()
device = "cpu"

def check_cuda():
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

def baseline(train_data, test_data):
    # Get the majority class
    all_data = train_data + test_data
    labels = [label for _, label in all_data]

    unique_labels, counts = np.unique(labels, return_counts=True)
    majority_class = unique_labels[np.argmax(counts)]
    logger.info(f"Majority class: {majority_class}")

    # Baseline: Always predict the majority class
    def baseline_predict(data):
        return [majority_class] * len(data)

    # Evaluate baseline accuracy
    true_labels = labels
    predicted_labels = baseline_predict(all_data)
    accuracy = np.mean([true == pred for true, pred in zip(true_labels, predicted_labels)])
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    logger.info(f"Baseline accuracy: {accuracy * 100:.2f}%")
    logger.info(f"Baseline precision: {precision * 100:.2f}%")
    logger.info(f"Baseline recall: {recall * 100:.2f}%")
    logger.info(f"Baseline F1-score: {f1 * 100:.2f}%")

    # For detailed per-class metrics
    logger.info(f"Detailed Classification Report: \n{classification_report(true_labels, predicted_labels)}")


def create_torch_dataloader(data, transform, batch_size=16, shuffle=False):
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

    dataset = TensorDataset(images_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def train_model(network, optimizer, loss_fn, num_epochs, enable_early_stopping=False, patience=5):
    torch.cuda.empty_cache()

    loss_values = []

    if enable_early_stopping:
        early_stopping = EarlyStopping(patience=patience, verbose=True)

    network.train()
    for epoch in tqdm(range(num_epochs), desc='Training model'):
        network.train()
        epoch_loss = 0.0
        num_batches = 0
        for images, target_labels in train_loader:
            images = images.to(device)
            target_labels = target_labels.to(device)

            pred_logits = network(images)
            loss = loss_fn(pred_logits, target_labels)
            epoch_loss += loss.item()
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_train_loss = epoch_loss / num_batches

        if enable_early_stopping:
            network.eval()
            val_loss = 0.0
            val_batches = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, target_labels in val_loader:
                    images = images.to(device)
                    target_labels = target_labels.to(device)
                    
                    pred_logits = network(images)
                    loss = loss_fn(pred_logits, target_labels)
                    val_loss += loss.item()
                    val_batches += 1
                    
                    _, predicted = torch.max(pred_logits, 1)
                    total += target_labels.size(0)
                    correct += (predicted == target_labels).sum().item()
            
            avg_val_loss = val_loss / val_batches
            val_accuracy = correct / total

        loss_values.append(avg_train_loss)
        
        if enable_early_stopping:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Early stopping check
        if enable_early_stopping:
            early_stopping(avg_val_loss, network)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                network.load_state_dict(early_stopping.best_model)
                break
    
    # Load best model
    if enable_early_stopping and early_stopping.best_model is not None:
        network.load_state_dict(early_stopping.best_model)
        logger.info("Loaded best model weights")

    logger.info(loss_values)
    

if __name__ == "__main__":
    print(torch.version.cuda)
    # Load the dataset
    preped_folder = os.path.join(config.DATA_DIR, "_preped")

    train_data = pd.read_csv(os.path.join(config.DATA_DIR, 'train_data.csv')).values.tolist()
    test_data = pd.read_csv(os.path.join(config.DATA_DIR, 'test_data.csv')).values.tolist()

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to consistent size
        transforms.ToTensor(),           # Convert to tensor [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    check_cuda()
    logger.info(f"Device set to: {device}")
    baseline(train_data, test_data)
    train_loader = create_torch_dataloader(train_data, transform, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = create_torch_dataloader(test_data, transform, batch_size=config.BATCH_SIZE, shuffle=False)
    logger.info(f"Train loader size: {len(train_loader.dataset)}")
    logger.info(f"Test loader size: {len(test_loader.dataset)}")



