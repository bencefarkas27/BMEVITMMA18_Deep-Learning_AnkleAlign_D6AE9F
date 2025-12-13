from utils import setup_logger, check_cuda, create_torch_dataloader, load_best_model, get_transforms
from networks import BestNetWork
import os
import pandas as pd
import config
import torch
import torchvision.transforms as transforms


logger = setup_logger()

def predict():
    preped_folder = os.path.join(config.DATA_DIR, "_preped")
    test_data = pd.read_csv(os.path.join(config.DATA_DIR, 'test_data.csv')).values.tolist()

    # Define image transformations
    transform = get_transforms()
    device = check_cuda(logger)
    logger.info(f"Device set to: {device}")
    test_loader = create_torch_dataloader(logger, test_data, preped_folder, transform, batch_size=config.BATCH_SIZE, shuffle=False)

    # Load the best cnn
    best_model_path = os.path.join(config.MODEL_DIR, 'best_cnn_weights.pth')
    best_model = load_best_model(best_model_path, device)
    logger.info(f"Best model loaded from {best_model_path}")
     
    # Perform predictions
    best_model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = best_model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # Convert predictions to labels
    label_mapping = {0: '1_Pronacio', 1: '2_Neutralis', 2: '3_Szupinacio'}
    predicted_labels = [label_mapping[pred] for pred in predictions]
    

    # Log predictions
    for i, (img_name, _) in enumerate(test_data):
        logger.info(f"Image: {img_name}, Predicted Label: {predicted_labels[i]}")

if __name__ == "__main__":
    predict()