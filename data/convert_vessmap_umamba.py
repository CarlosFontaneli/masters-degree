import os
import shutil
import json
import random
from PIL import Image
import numpy as np

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

# Process training cases
for img_file in train_files:
    case_id = os.path.splitext(img_file)[0]

    src_img_path = os.path.join(images_source, img_file)
    src_label_path = os.path.join(labels_source, case_id + '.png')

    target_img_name = f"{case_id}_0000.png"
    target_label_name = f"{case_id}.png"

    # Open the image, convert to grayscale ('L') and save
    image = Image.open(src_img_path).convert('L')
    image.save(os.path.join(imagesTr_dir, target_img_name))

    if os.path.exists(src_label_path):
        # Open label, convert to grayscale, and normalize (set nonzero to 1)
        label_img = Image.open(src_label_path).convert('L')
        label_arr = np.array(label_img)
        label_arr[label_arr > 0] = 1
        normalized_label = Image.fromarray(label_arr.astype(np.uint8))
        normalized_label.save(os.path.join(labelsTr_dir, target_label_name))
    else:
        print(f"Warning: Label not found for training case {case_id}")

# Process validation cases
for img_file in test_files:
    case_id = os.path.splitext(img_file)[0]

    src_img_path = os.path.join(images_source, img_file)
    src_label_path = os.path.join(labels_source, case_id + '.png')

    target_img_name = f"{case_id}_0000.png"
    target_label_name = f"{case_id}.png"

    image = Image.open(src_img_path).convert('L')
    image.save(os.path.join(imagesTs_dir, target_img_name))

    if os.path.exists(src_label_path):
        label_img = Image.open(src_label_path).convert('L')
        label_arr = np.array(label_img)
        label_arr[label_arr > 0] = 1
        normalized_label = Image.fromarray(label_arr.astype(np.uint8))
        normalized_label.save(os.path.join(labelsTs_dir, target_label_name))
    else:
        print(f"Warning: Label not found for validation case {case_id}")

# Create dataset.json metadata file
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
