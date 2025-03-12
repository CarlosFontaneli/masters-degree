#!/usr/bin/env python3
import os
import sys
import json
import datetime
import argparse
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import monai

sys.path.append('../data/vess-map')
from vess_map_dataset import VessMapDataset

# Import the UMamba architecture helper
sys.path.append('/home/fonta42/Desktop/masters-degree/U-Mamba')
from umamba.nnunetv2.nets.UMambaEnc_2d import UMambaEnc
from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op


# -------------------------------
# Logging Utilities
# -------------------------------
LOG_RECORDS = []

def log_message(level, message):
    logger = logging.getLogger("UMamba_Training")
    logger.log(level, message)
    LOG_RECORDS.append({
        "level": logging.getLevelName(level),
        "message": message,
        "time": datetime.datetime.now().isoformat()
    })

# -------------------------------
# Dice / Metrics
# -------------------------------
def compute_dice(preds, targets):
    preds = preds.long()
    targets = targets.long()
    intersection = (preds & targets).sum(dim=(2, 3))
    dice = (2.0 * intersection.float()) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-6)
    return dice.mean().item()

# -------------------------------
# Visualization Utilities
# -------------------------------
def create_difference_mask(bin_preds, bin_targets):
    b, _, h, w = bin_preds.shape
    diff_masks = np.zeros((b, h, w, 3), dtype=np.uint8)
    for idx in range(b):
        bp = bin_preds[idx, 0]
        bt = bin_targets[idx, 0]
        tp = (bp == 1) & (bt == 1)
        fp = (bp == 1) & (bt == 0)
        fn = (bp == 0) & (bt == 1)
        diff_masks[idx][tp] = [0, 255, 0]       # Green for true positive
        diff_masks[idx][fp] = [255, 255, 0]       # Yellow for false positive
        diff_masks[idx][fn] = [255, 0, 0]         # Red for false negative
    return diff_masks

def save_comparison_plot(original_img, bin_pred, diff_mask, epoch, batch_idx, model_name):
    if torch.is_tensor(original_img):
        original_img = original_img.cpu().numpy()
    if torch.is_tensor(bin_pred):
        bin_pred = bin_pred.cpu().numpy()
    original_img = np.transpose(original_img, (1, 2, 0))
    output_dir = f"./u-mamba/validation-diff-masks/{model_name}_validation-diff-masks"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(original_img, cmap="gray")
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    axs[1].imshow(bin_pred, cmap="gray")
    axs[1].set_title("Prediction")
    axs[1].axis("off")
    
    axs[2].imshow(diff_mask)
    axs[2].set_title("Diff Mask")
    axs[2].axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"comparison_epoch{epoch}_batch{batch_idx}.png")
    plt.savefig(save_path)
    plt.close(fig)

# -------------------------------
# Loss Functions and Helper
# -------------------------------
def compute_loss(preds, masks, seg_loss_func, ce_loss_func, loss_type):
    if loss_type == "dice":
        return seg_loss_func(preds, masks)
    elif loss_type == "ce":
        return ce_loss_func(preds, masks.float())
    elif loss_type == "both":
        return seg_loss_func(preds, masks) + ce_loss_func(preds, masks.float())
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# -------------------------------
# Training and Validation Loops
# -------------------------------
def train_one_epoch(model, dataloader, device, optimizer, seg_loss_func, ce_loss_func, loss_type, accumulate_grad_steps=1, clip_grad=True):
    model.train()
    running_loss = 0.0
    step_count = 0
    for step, (images, masks, skeletons) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        masks = masks.to(device)
        preds = model(images)
        loss = compute_loss(preds, masks, seg_loss_func, ce_loss_func, loss_type)
        loss.backward()
        if (step + 1) % accumulate_grad_steps == 0:
            if clip_grad:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        step_count += 1
    return running_loss / step_count

def validate(model, dataloader, device, seg_loss_func, ce_loss_func, loss_type, epoch=0, model_name="umamba_model"):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, (images, masks, skeletons) in enumerate(tqdm(dataloader, desc="Validating")):
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            loss = compute_loss(preds, masks, seg_loss_func, ce_loss_func, loss_type)
            val_loss += loss.item()
            probs = torch.sigmoid(preds)
            bin_preds = (probs > 0.5).float()
            bin_masks = (masks > 0.5).float()
            dice_val = compute_dice(bin_preds, bin_masks)
            val_dice += dice_val
            count += 1
            # Optionally save a diff mask every N epochs
            if epoch % 50 == 0 and epoch != 0 and batch_idx == 0:
                diff_masks = create_difference_mask(bin_preds.cpu().numpy(), bin_masks.cpu().numpy())
                save_comparison_plot(images[0], bin_preds[0, 0], diff_masks[0], epoch, batch_idx, model_name)
    return val_loss / count, val_dice / count

# -------------------------------
# Simple Configuration for UMamba Model
# -------------------------------
input_size = (256, 256)            # Input patch size
input_channels = 3                 # RGB images
n_stages = 4                       # Number of stages in the encoder/decoder
features_per_stage = [32, 64, 128, 256]  # Feature channels per stage
kernel_sizes = [[3, 3]] * n_stages       # 3x3 convolutions at each stage
# Define strides (e.g., no downsampling at first stage, then downsample by factor of 2)
strides = [[1, 1], [2, 2], [2, 2], [2, 2]]
n_conv_per_stage = 2               # Number of convolutional blocks per encoder stage
n_conv_per_stage_decoder = 2       # Number of convolutional blocks per decoder stage
num_classes = 1                    # For binary segmentation
conv_op = nn.Conv2d               # Use standard 2D convolutions
norm_op = nn.InstanceNorm2d       # InstanceNorm is commonly used
norm_op_kwargs = {'eps': 1e-5, 'affine': True}
nonlin = nn.LeakyReLU             # Activation function
nonlin_kwargs = {'inplace': True}
deep_supervision = False           # If you want deep supervision from the decoder
stem_channels = features_per_stage[0]  # The stem uses the first stage's feature count

# -------------------------------
# Main Training Routine for UMamba
# -------------------------------
def train_umamba(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(logging.INFO, f"Using device: {device}")

    log_message(logging.INFO, "Preparing dataset...")
    vess_dataset = VessMapDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        skeleton_dir=args.skeleton_dir,
        image_size=args.image_size,
        apply_transform=args.augment
    )
    train_loader, test_loader = vess_dataset.vess_map_dataloader(
        batch_size=args.batch_size, train_size=args.train_size / 100
    )

    # Define loss functions
    class_weight_tensor = vess_dataset.class_weights_tensor
    ce_loss_func = nn.BCEWithLogitsLoss(pos_weight=class_weight_tensor[1]/class_weight_tensor[0], reduction="mean")
    seg_loss_func = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

    log_message(logging.INFO, "Initializing UMamba model...")

   # Create the U-Mamba model instance directly:
    model = UMambaEnc(
        input_size=input_size,
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=norm_op,
        norm_op_kwargs=norm_op_kwargs,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nonlin,
        nonlin_kwargs=nonlin_kwargs,
        deep_supervision=deep_supervision,
        stem_channels=stem_channels
    ).to(device)
    
    log_message(logging.INFO, f"UMamba Model Loaded.")

    # Choose optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unknown optimizer. Choose either 'sgd' or 'adam'.")

    # Choose scheduler if needed (here we use cosine annealing)
    if args.scheduler == "cosine":
        num_steps = args.epochs * len(train_loader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    else:
        scheduler = None

    log_message(logging.INFO, "Training parameters:")
    for arg_name, val in vars(args).items():
        log_message(logging.INFO, f" - {arg_name}: {val}")

    now = datetime.datetime.now()
    date_str = now.strftime("%d%m%Y")
    model_name = f"umamba_{args.optimizer}_lr{args.lr}_bs{args.batch_size}_{args.loss_type}_{args.train_size}pct_{date_str}"

    train_losses = []
    val_losses = []
    val_dices = []
    best_loss = float("inf")

    for epoch in range(args.epochs):
        log_message(logging.INFO, f"Starting epoch {epoch+1}/{args.epochs}")
        
        # Record start time and VRAM usage (if CUDA)
        epoch_start = time.time()
        if device.type == "cuda":
            vram_before = torch.cuda.memory_allocated(device)
            
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            seg_loss_func=seg_loss_func,
            ce_loss_func=ce_loss_func,
            loss_type=args.loss_type,
            accumulate_grad_steps=args.accumulate_grad_steps,
            clip_grad=True
        )
        
        epoch_time = time.time() - epoch_start
        if device.type == "cuda":
            vram_after = torch.cuda.memory_allocated(device)
            log_message(logging.INFO, 
                        f"Epoch {epoch+1} training time: {epoch_time:.2f} sec, VRAM before: {vram_before/1e6:.2f} MB, after: {vram_after/1e6:.2f} MB")
        else:
            log_message(logging.INFO, f"Epoch {epoch+1} training time: {epoch_time:.2f} sec")
        
        train_losses.append(train_loss)

        val_loss, val_dice = validate(
            model=model,
            dataloader=test_loader,
            device=device,
            seg_loss_func=seg_loss_func,
            ce_loss_func=ce_loss_func,
            loss_type=args.loss_type,
            epoch=epoch,
            model_name=model_name
        )
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        log_message(logging.INFO, f"Epoch {epoch+1}, TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, ValDice={val_dice:.4f}")

        if scheduler is not None:
            scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs("./u-mamba/models", exist_ok=True)
            torch.save(model.state_dict(), f"./u-mamba/models/{model_name}_best.pth")
            log_message(logging.INFO, f"Best model saved at epoch {epoch+1} with val_loss {best_loss:.4f}")

        torch.save(model.state_dict(), f"./u-mamba/models/{model_name}_latest.pth")

    # Save metrics and plot results
    metrics = {"train_losses": train_losses, "val_losses": val_losses, "val_dices": val_dices}
    os.makedirs("./u-mamba/metrics", exist_ok=True)
    with open(f"./u-mamba/metrics/{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    log_message(logging.INFO, f"Metrics saved to ./u-mamba/metrics/{model_name}_metrics.json")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    l1 = ax1.plot(range(1, args.epochs+1), train_losses, label="Train Loss", color="tab:blue", linestyle="-")
    l2 = ax1.plot(range(1, args.epochs+1), val_losses, label="Val Loss", color="tab:blue", linestyle="--")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Val Dice", color="tab:red")
    l3 = ax2.plot(range(1, args.epochs+1), val_dices, label="Val Dice", color="tab:red", linestyle="-")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center")
    plt.title("Training & Validation Loss and Dice Over Epochs")
    plt.grid(True)
    os.makedirs("./u-mamba/metrics", exist_ok=True)
    plt.savefig(f"./u-mamba/metrics/{model_name}_training_validation_metrics.png")
    plt.close(fig)
    log_message(logging.INFO, f"Plot saved to ./u-mamba/metrics/{model_name}_training_validation_metrics.png")

    os.makedirs("./u-mamba/logs", exist_ok=True)
    with open(f"./u-mamba/logs/{model_name}_logs.json", "w") as f:
        json.dump(LOG_RECORDS, f, indent=4)
    log_message(logging.INFO, f"Logs saved to ./u-mamba/logs/{model_name}_logs.json")

# -------------------------------
# Argument Parsing
# -------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train a UMamba model on the VessMapDataset.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=80, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"], help="Optimizer choice")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (for SGD optimizer)")
    parser.add_argument("--loss_type", type=str, default="both", choices=["dice", "ce", "both"], help="Loss type")
    parser.add_argument("--scheduler", type=str, default="none", choices=["cosine", "none"], help="Scheduler type")
    parser.add_argument("--train_size", type=float, default=80, help="Train/validation split ratio as a percentage")
    parser.add_argument("--image_size", type=int, default=256, help="Final cropped image size for data augmentation")
    parser.add_argument("--accumulate_grad_steps", type=int, default=1, help="Accumulate grad steps before optimizer step")
    parser.add_argument("--image_dir", type=str, default="../data/vess-map/images", help="Images directory")
    parser.add_argument("--mask_dir", type=str, default="../data/vess-map/labels", help="Labels directory")
    parser.add_argument("--skeleton_dir", type=str, default="../data/vess-map/skeletons", help="Skeleton directory")
    parser.add_argument("--augment", type=bool, default=True, help="Apply random data augmentation")
    return parser.parse_args()

if __name__ == "__main__":
    logger = logging.getLogger("UMamba_Training")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    args = get_args()
    train_umamba(args)
    log_message(logging.INFO, "Training completed.")
