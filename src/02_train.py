import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchinfo import summary

from tqdm.auto import tqdm

from utils import setup_logger, check_cuda, create_torch_dataloader, get_transforms
from networks import BaseLineNetWork, BestNetWork
import config

logger = setup_logger()


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
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

def train_model(device, network, optimizer, loss_fn, num_epochs, train_loader, val_loader=None, patience=5):
    torch.cuda.empty_cache()

    loss_values = []

    if val_loader is not None:
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

        if val_loader is not None:
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
        
        if val_loader is not None:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Early stopping check
        if val_loader is not None:
            early_stopping(avg_val_loss, network)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                network.load_state_dict(early_stopping.best_model)
                break
    
    # Load best model
    if val_loader is not None and early_stopping.best_model is not None:
        network.load_state_dict(early_stopping.best_model)
        logger.info("Loaded best model weights")

    logger.info(loss_values)
    
def main():
    # Load the dataset
    preped_folder = os.path.join(config.DATA_DIR, "_preped")

    train_data = pd.read_csv(os.path.join(config.DATA_DIR, 'train_data.csv')).values.tolist()

    # Define image transformations
    transform = get_transforms()

    device = check_cuda(logger)
    logger.info(f"Device set to: {device}")

    # Create DataLoader for baseline model
    train_loader = create_torch_dataloader(logger, train_data, preped_folder, transform, batch_size=config.BATCH_SIZE, shuffle=True)
    basline_cnn = BaseLineNetWork().to(device)
    logger.info(summary(basline_cnn, input_size=(config.BATCH_SIZE, 1, 224, 224)))

    # Train the baseline cnn
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(basline_cnn.parameters(), lr=10*config.LEARNING_RATE)
    train_model(device, basline_cnn, optimizer, loss_fn, config.NUM_EPOCHS, train_loader)

    # Create models directory if it doesn't exist
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    # Save the baseline model's weights
    model_path = os.path.join(config.MODEL_DIR, "baseline_weights.pth")
    torch.save(basline_cnn.state_dict(), model_path)
    logger.info(f"Baseline model weights saved to {model_path}")

    # Train/validation split for best model
    transform = get_transforms(data_augmentation=True)
    train_loader, val_loader = create_torch_dataloader(logger, train_data, preped_folder, transform, batch_size=config.BATCH_SIZE, shuffle=True, val_split=0.2)
    logger.info(f"Train loader size: {len(train_loader.dataset)}")
    logger.info(f"Validation loader size: {len(val_loader.dataset)}")

    best_cnn = BestNetWork().to(device)
    best_cnn.apply(init_weights)
    logger.info(summary(best_cnn, input_size=(config.BATCH_SIZE, 1, 224, 224)))

    # Train the model
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(best_cnn.parameters(), lr=config.LEARNING_RATE)
    train_model(device, best_cnn, optimizer, loss_fn, config.NUM_EPOCHS, train_loader, val_loader=val_loader, patience=config.EARLY_STOPPING_PATIENCE)

    # Save the trained model's weights
    model_path = os.path.join(config.MODEL_DIR, "best_cnn_weights.pth")
    torch.save(best_cnn.state_dict(), model_path)
    logger.info(f"Best CNN weights saved to {model_path}")

if __name__ == "__main__":
    main()


