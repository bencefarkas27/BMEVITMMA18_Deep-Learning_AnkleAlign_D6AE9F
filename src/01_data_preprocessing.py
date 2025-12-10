import numpy as np
import os
import shutil
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from collections import Counter
import logging
import sys
from utils import setup_logger
import config

logger = setup_logger()

def rename_images(data_source_folder):
    # Rename images to avoid name conflicts and copy to _preped folder
    image_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    preped_folder = os.path.join(config.DATA_DIR, '_preped')
    os.makedirs(preped_folder, exist_ok=True)

    for foldername in os.listdir(data_source_folder):
        folder_path = os.path.join(data_source_folder, foldername)
        if os.path.isdir(folder_path) and foldername != 'consensus':
            logger.info(f"Renaming files in folder: {foldername}")
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    name, extension = os.path.splitext(filename)
                    if extension in image_formats:
                        new_filename = f"{foldername}_{name}{extension}"
                        new_file_path = os.path.join(preped_folder, new_filename)
                        shutil.copy2(file_path, new_file_path)
    return preped_folder

def remove_random_hash_from_labelfiles(data_source_folder):
    label_file = ''
    # Create labels folder and save modified JSON
    labels_folder = os.path.join(config.DATA_DIR, '_label_files')
    os.makedirs(labels_folder, exist_ok=True)
    for foldername in os.listdir(data_source_folder):
        folder_path = os.path.join(data_source_folder, foldername)
        if os.path.isdir(folder_path) and foldername != 'consensus': 
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    name, extension = os.path.splitext(filename)
                    if extension == '.json':
                        label_file = file_path
                        break
                    else:
                        label_file = 'not found'
            if label_file == 'not found':
                logger.info(f"No label file found in folder: {foldername}")
                continue

            logger.info(f"Label file found: {label_file}")
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cnt = 0
            # Process each entry
            for entry in data:
                if 'file_upload' in entry:
                    # Remove labelstudio hash prefix
                    parts = entry['file_upload'].split('-', 1)  # Split only on first '-'
                    if len(parts) > 1:
                        entry['file_upload'] = f"{foldername}_{parts[1]}"
                        cnt += 1

            # Get original filename and create new path
            original_filename = os.path.basename(label_file)
            new_label_path = os.path.join(labels_folder, original_filename)

            # Save the modified JSON
            with open(new_label_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"Total entries updated: {cnt}")
    return labels_folder

def match_images_and_labels(labels_folder, preped_folder):
    #Match the file names with the labels
    image_names = list(os.listdir(preped_folder))
    data_ready = []
    for label_filename in os.listdir(labels_folder):
        label_path = os.path.join(labels_folder, label_filename)
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        for entry in labels:
            if 'file_upload' in entry:
                if entry['file_upload'] in image_names:
                    result = entry['annotations'][0].get('result')
                    if len(result) > 0:
                        label = result[0].get('value').get('choices')[0]
                        data_ready.append((entry['file_upload'], label))
    logger.info(f"Total matched entries: {len(data_ready)}")

    # Reaname the 3 wrong labels:
    for i in range(len(data_ready)):
        imge_name, label = data_ready[i]
        if label == 'neutral': data_ready[i] = (imge_name, '2_Neutralis')
        elif label == 'pronation': data_ready[i] = (imge_name, '1_Pronacio')
        elif label == 'supination': data_ready[i] = (imge_name, '3_Szupinacio')
    
    return data_ready

def process_consensus_folder(data_source_folder):
    # Read what picture are in consensus text file
    consensus_file_path =  os.path.join(data_source_folder, 'consensus')
    consensus_file = os.path.join(consensus_file_path, 'anklealign-consensus.txt')
    with open(consensus_file, 'r', encoding='utf-8') as f:
        consensus_images = f.read().splitlines()

    img_names = []
    # Get every image name from the consensus file
    for img in consensus_images:
        parts = img.split('\\')
        if len(parts) > 1:
            img_names.append(parts[-1])

    # Count occurrences of each image name
    img_counts = Counter(img_names)

    # Keep only images that appear exactly once
    unique_consensus_image_names = [img for img, count in img_counts.items() if count == 1]

    logger.info(f"Total images in consensus: {len(img_names)}")
    logger.info(f"Unique images (appearing exactly once): {len(unique_consensus_image_names)}")
    logger.info(f"Duplicate images removed: {len(img_names) - len(unique_consensus_image_names)}")

    consensus_label_matrix = pd.DataFrame({
        'image': unique_consensus_image_names,
        '1_Pronacio': 0,
        '2_Neutralis': 0,
        '3_Szupinacio': 0
    })

    for consensus_label_file in os.listdir(consensus_file_path):
        extension = os.path.splitext(consensus_label_file)[1]
        consensus_label_path = os.path.join(consensus_file_path, consensus_label_file)
        if extension != '.json' or os.path.getsize(consensus_label_path) == 0:
            continue

        with open(consensus_label_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        for entry in labels:
            if 'file_upload' in entry:
                img_name = entry['file_upload'].split('-', 1)[1]  # Remove hash prefix
                if img_name in unique_consensus_image_names:
                    result = entry['annotations'][0].get('result')
                    if len(result) > 0:
                        label = result[0].get('value').get('choices')[0]
                        if label == '1_Pronacio':
                            consensus_label_matrix.loc[consensus_label_matrix['image'] == img_name, '1_Pronacio'] += 1
                        elif label == '2_Neutralis':
                            consensus_label_matrix.loc[consensus_label_matrix['image'] == img_name, '2_Neutralis'] += 1
                        elif label == '3_Szupinacio':
                            consensus_label_matrix.loc[consensus_label_matrix['image'] == img_name, '3_Szupinacio'] += 1

        unique_consensus_images = []

    # Rename the images as the prepared data
    for img in consensus_images:
        parts = img.split('\\')
        if len(parts) > 1 and parts[2] in unique_consensus_image_names:
            row = consensus_label_matrix.loc[consensus_label_matrix['image'] == parts[2]]
            max_col = row[['1_Pronacio', '2_Neutralis', '3_Szupinacio']].idxmax(axis=1).values[0]
            label = max_col
            unique_consensus_images.append((f"{parts[1]}_{parts[2]}", label))
    logger.info(f"Total unique consensus images: {len(unique_consensus_images)}")
    return unique_consensus_images

def separate_train_test(img_label_pairs, unique_consensus_images):
    # Match the consensus images with the prepared data
    matched_consensus = []
    for img, _ in unique_consensus_images:
        for data_img, _ in img_label_pairs:
            if img == data_img:
                matched_consensus.append((data_img))

    train_data = [(img, label) for img, label in img_label_pairs if img not in matched_consensus]
    test_data = unique_consensus_images

    logger.info(f"Total training data: {len(train_data)}")
    logger.info(f"Total testing data: {len(test_data)}")

    return train_data, test_data

def preprocess():
    data_source_folder = os.path.join(config.DATA_DIR, 'anklealign')
    preped_folder = rename_images(data_source_folder)
    labels_folder = remove_random_hash_from_labelfiles(data_source_folder)
    img_label_pairs = match_images_and_labels(labels_folder, preped_folder)
    unique_consensus_images = process_consensus_folder(data_source_folder)
    # Remove unique consensus images from img_label_pairs, because consensus data will be used as test set
    train_data, test_data = separate_train_test(img_label_pairs, unique_consensus_images)

    # Save train and test data to files
    pd.DataFrame(train_data, columns=['image', 'label']).to_csv(os.path.join(config.DATA_DIR, 'train_data.csv'), index=False)
    pd.DataFrame(test_data, columns=['image', 'label']).to_csv(os.path.join(config.DATA_DIR, 'test_data.csv'), index=False)

if __name__ == "__main__":
    preprocess()

