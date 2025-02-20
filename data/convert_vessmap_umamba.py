#!/usr/bin/env python3
import os
import shutil
import json
import random

raw_data_dir = '/home/fonta42/Desktop/masters-degree/data/vess-map'
target_dir = '/home/fonta42/Desktop/masters-degree/data/vess-map-umamba'
train_ratio = 0.8

images_source = os.path.join(raw_data_dir, 'png_images')
labels_source = os.path.join(raw_data_dir, 'labels')

imagesTr_dir = os.path.join(target_dir, 'imagesTr')
labelsTr_dir = os.path.join(target_dir, 'labelsTr')
imagesTs_dir = os.path.join(target_dir, 'imagesTs')
labelsTs_dir = os.path.join(target_dir, 'labelsTs')  

os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)
os.makedirs(imagesTs_dir, exist_ok=True)
os.makedirs(labelsTs_dir, exist_ok=True)

png_exts = '.png'
all_image_files = [f for f in os.listdir(images_source) if f.lower().endswith(png_exts)]
all_image_files.sort()

# Randomize and split the cases using train_ratio
random.shuffle(all_image_files)
num_total = len(all_image_files)
num_train = int(train_ratio * num_total)
train_files = all_image_files[:num_train]
test_files = all_image_files[num_train:]

print(f"Total cases: {num_total}, Training: {len(train_files)}, Validation: {len(test_files)}")

for img_file in train_files:
    # Use the filename (without extension) as the unique case identifier
    case_id = os.path.splitext(img_file)[0]
    
    src_img_path = os.path.join(images_source, img_file)
    src_label_path = os.path.join(labels_source, case_id + '.png')
    
    # For nnU-Net, single-channel images are stored as {CASE_ID}_0000.<ext>
    target_img_name = f"{case_id}_0000.png"
    target_label_name = f"{case_id}.png"
    
    shutil.copy(src_img_path, os.path.join(imagesTr_dir, target_img_name))
    
    if os.path.exists(src_label_path):
        shutil.copy(src_label_path, os.path.join(labelsTr_dir, target_label_name))
    else:
        print(f"Warning: Label not found for training case {case_id}")

for img_file in test_files:
    case_id = os.path.splitext(img_file)[0]
    
    src_img_path = os.path.join(images_source, img_file)
    src_label_path = os.path.join(labels_source, case_id + '.png')
    
    target_img_name = f"{case_id}_0000.png"
    target_label_name = f"{case_id}.png"
    
    shutil.copy(src_img_path, os.path.join(imagesTs_dir, target_img_name))
    
    if os.path.exists(src_label_path):
        shutil.copy(src_label_path, os.path.join(labelsTs_dir, target_label_name))
    else:
        print(f"Warning: Label not found for validation case {case_id}")

# This file contains metadata needed by nnU-Net. Note that "numTraining" refers to the number of training cases
dataset_json = {
    "channel_names": {
        "0": "Grayscale"
    },
    "labels": {
        "background": 0,
        "vessel": 1
    },
    "numTraining": len(train_files),
    "file_ending": ".png"
}

with open(os.path.join(target_dir, 'dataset.json'), 'w') as f:
    json.dump(dataset_json, f, indent=4)

print("Dataset conversion complete!")
