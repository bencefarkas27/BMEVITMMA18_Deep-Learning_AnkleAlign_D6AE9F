import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

import config
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, balanced_accuracy_score

from networks import BaseLineNetWork, BestNetWork
from utils import setup_logger, check_cuda, create_torch_dataloader, load_baseline_model, load_best_model, get_transforms

logger = setup_logger()

def stat_baseline(train_data, test_data):
    logger.info("Calculating baseline (majority class) performance metrics")
    # Get the majority class from training data
    labels = [label for _, label in train_data]

    unique_labels, counts = np.unique(labels, return_counts=True)
    majority_class = unique_labels[np.argmax(counts)]
    logger.info(f"Majority class: {majority_class}")

    # Baseline: Always predict the majority class
    def baseline_predict(data):
        return [majority_class] * len(data)

    # Evaluate baseline accuracy
    true_labels = [label for _, label in test_data]
    predicted_labels = baseline_predict(test_data)
    balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    logger.info(f"Baseline balanced accuracy: {balanced_acc * 100:.2f}%")
    logger.info(f"Baseline precision: {precision * 100:.2f}%")
    logger.info(f"Baseline recall: {recall * 100:.2f}%")
    logger.info(f"Baseline F1-score: {f1 * 100:.2f}%")

    # For detailed per-class metrics
    logger.info(f"Detailed Classification Report: \n{classification_report(true_labels, predicted_labels)}")

def evaluate(model, model_name, test_loader, device):
    logger.info(f"Evaluating {model_name} on consensus test set")
    true_labels = test_loader.dataset.tensors[1].numpy()
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.cpu().numpy())

    balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    logger.info(f"network balanced accuracy: {balanced_acc * 100:.2f}%")
    logger.info(f"network precision: {precision * 100:.2f}%")
    logger.info(f"network recall: {recall * 100:.2f}%")
    logger.info(f"network F1 score: {f1 * 100:.2f}%")

    logger.info(f"Detailed Classification Report: \n{classification_report(true_labels, predicted_labels)}")


def main():
    # Load the dataset
    preped_folder = os.path.join(config.DATA_DIR, "_preped")
    train_data_for_baseline = pd.read_csv(os.path.join(config.DATA_DIR, 'train_data.csv')).values.tolist()
    test_data = pd.read_csv(os.path.join(config.DATA_DIR, 'test_data.csv')).values.tolist()

    stat_baseline(train_data_for_baseline, test_data)

    # Define image transformations
    transform = get_transforms()
    device = check_cuda(logger)
    logger.info(f"Device set to: {device}")

    test_loader = create_torch_dataloader(logger, test_data, preped_folder, transform, batch_size=config.BATCH_SIZE, shuffle=False)
    logger.info(f"Test loader size: {len(test_loader.dataset)}")

    # Load the baseline cnn
    model_path = os.path.join(config.MODEL_DIR, 'baseline_weights.pth')
    model = load_baseline_model(model_path, device)
    logger.info(f"Model loaded from {model_path}")
    evaluate(model, "Baseline CNN", test_loader, device)

    # Load the best cnn
    best_model_path = os.path.join(config.MODEL_DIR, 'best_cnn_weights.pth')
    best_model = load_best_model(best_model_path, device)
    logger.info(f"Best model loaded from {best_model_path}")
    evaluate(best_model, "Best CNN", test_loader, device)

if __name__ == "__main__":
    main()