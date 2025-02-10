import os
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import torch

class VessMapDataset(Dataset):
    def __init__(self, image_dir, mask_dir, skeleton_dir, mode='train', apply_transform=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.skeleton_dir = skeleton_dir
        self.mode = mode
        self.apply_transform_flag = apply_transform
        self.class_weights_tensor = torch.tensor([0.26, 0.74])

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

    def apply_transform(self, image, mask, skeleton, seed=42):
        # Always resize to 1024x1024
        resize = transforms.Resize((1024, 1024))
        image = resize(image)
        mask = resize(mask)
        skeleton = resize(skeleton)

        if self.apply_transform_flag and self.mode == 'train':
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

        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        skeleton = TF.to_tensor(skeleton)

        return image, mask, skeleton

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.labels[idx]
        skeleton = self.skeletons[idx]

        image, mask, skeleton = self.apply_transform(image, mask, skeleton)

        # Get image dimensions
        _, height, width = image.shape  # image shape is (C, H, W)

        # Generate bounding box that covers the full image
        # (x_min, y_min, x_max, y_max)
        bbox = torch.tensor([0, 0, width, height], dtype=torch.float32)

        return image, mask, skeleton, bbox

    def vess_map_dataloader(self, batch_size, train_size, shuffle=True):
        dataset_size = len(self)
        train_len = int(train_size * dataset_size)
        val_len = dataset_size - train_len

        # Split indices into train and test
        indices = list(range(dataset_size))
        train_indices = indices[:train_len]
        test_indices = indices[train_len:]

        # Create separate instances of the dataset for train and test
        train_dataset = VessMapDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            skeleton_dir=self.skeleton_dir,
            mode='train',  # Set mode to 'train' for the training dataset
            apply_transform=self.apply_transform_flag
        )

        test_dataset = VessMapDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            skeleton_dir=self.skeleton_dir,
            mode='test',  # Set mode to 'test' for the test dataset
            apply_transform=False  # Ensure no transformations are applied to the test dataset
        )

        # Use the indices to create subsets
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

        # Create DataLoaders for train and test datasets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader