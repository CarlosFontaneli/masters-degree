import os
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import torch

class VessMapDataset(Dataset):
    def __init__(self, image_dir, mask_dir, skeleton_dir, image_size, apply_transform=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.skeleton_dir = skeleton_dir
        self.image_size = image_size
        self.apply_transform_flag = apply_transform

        # List all image files with .tiff or .tif extension
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.tiff') or f.endswith('.tif')]
        self.image_files.sort()

        # Find pairs of images, masks, and skeletons with matching base names
        self.pairs = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            mask_file = base_name + ".png"
            skeleton_file = base_name + ".png"

            mask_path = os.path.join(self.mask_dir, mask_file)
            skeleton_path = os.path.join(self.skeleton_dir, skeleton_file)

            if os.path.exists(mask_path) and os.path.exists(skeleton_path):
                self.pairs.append((img_file, mask_file, skeleton_file))
            else:
                print(f"Warning: Missing mask or skeleton for image {img_file}")

        # Load images, masks, and skeletons into lists
        self.images = []
        self.labels = []
        self.skeletons = []

        for img_file, mask_file, skeleton_file in self.pairs:
            image_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)
            skeleton_path = os.path.join(self.skeleton_dir, skeleton_file)

            # Open images
            image = Image.open(image_path).convert('RGB')  # Convert images to RGB
            mask = Image.open(mask_path).convert('L')      # Masks are grayscale
            skeleton = Image.open(skeleton_path).convert('L')  # Skeletons are grayscale

            self.images.append(image)
            self.labels.append(mask)
            self.skeletons.append(skeleton)

    def __len__(self):
        return len(self.pairs)

    def apply_transform(self, image, mask, skeleton):
        # For consistent transformations
        seed = random.randint(0, 2**32)
        random.seed(seed)
        torch.manual_seed(seed)

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            skeleton = TF.hflip(skeleton)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            skeleton = TF.vflip(skeleton)

        # Random rotation
        angle = random.uniform(-30, 30)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        skeleton = TF.rotate(skeleton, angle)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.image_size, self.image_size)
        )
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        skeleton = TF.crop(skeleton, i, j, h, w)

        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        skeleton = TF.to_tensor(skeleton)

        return image, mask, skeleton

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.labels[idx]
        skeleton = self.skeletons[idx]

        # Apply transformations
        if self.apply_transform_flag:
            image, mask, skeleton = self.apply_transform(image, mask, skeleton)
        else:
            # Convert to tensor without applying random transforms
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)
            skeleton = TF.to_tensor(skeleton)

        return image, mask, skeleton


    def vess_map_dataloader(
        self, batch_size, train_size, shuffle=True
    ):
        dataset_size = len(self)
        train_len = int(train_size * dataset_size)
        val_len = dataset_size - train_len

        # Split the current dataset into train and validation/test datasets
        train_dataset, test_dataset = random_split(self, [train_len, val_len])

        # Create DataLoaders for train and test datasets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
