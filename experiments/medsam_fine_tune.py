import sys
import os
import json
import datetime
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import monai
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import lightning as L
from lightning.fabric import Fabric

sys.path.append('../data/vess-map')
sys.path.append('../MedSAM')

from vess_map_dataset_medsam import VessMapDataset
from segment_anything import sam_model_registry

# -------------------------------
# Global Log Storage
# -------------------------------
LOG_RECORDS = []

def log_message(level, message):
    logger = logging.getLogger("MedSAM_Training")
    logger.log(level, message)
    LOG_RECORDS.append({
        "level": logging.getLevelName(level),
        "message": message,
        "time": datetime.datetime.now().isoformat()
    })

# -------------------------------
# IoU / Metrics
# -------------------------------
def compute_iou(preds, targets):
    preds = preds.long()
    targets = targets.long()
    intersection = (preds & targets).sum(dim=(2,3))
    union = (preds | targets).sum(dim=(2,3))
    iou = (intersection.float() / (union.float() + 1e-6)).mean()
    return iou.item()

# -------------------------------
# Loss Combination
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
# Train One Epoch
# -------------------------------
def train_one_epoch(model, dataloader, device, optimizer,
                    seg_loss_func, ce_loss_func, loss_type,
                    accumulate_grad_steps=1, clip_grad=True,
                    fabric=None):
    model.train()
    running_loss = 0.0
    step_count = 0

    for step, (images, masks, skeletons, bboxes) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        masks = masks.to(device)
        bboxes_np = bboxes.cpu().numpy()

        preds = model(images, bboxes_np)
        loss = compute_loss(preds, masks, seg_loss_func, ce_loss_func, loss_type)

        if fabric is not None:
            fabric.backward(loss)
        else:
            loss.backward()

        if (step + 1) % accumulate_grad_steps == 0:
            if clip_grad:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        step_count += 1

    return running_loss / step_count

# -------------------------------
# Visualization of Differences
# -------------------------------
def save_difference_mask(bin_preds, bin_targets, epoch, batch_idx, save_dir="diff_masks"):
    os.makedirs(save_dir, exist_ok=True)

    # Extract the first sample
    bin_pred = bin_preds[0, 0]  # shape (H, W)
    bin_tgt = bin_targets[0, 0] # shape (H, W)

    # TP == (1,1), FP == (1,0), FN == (0,1)
    # Build a (H, W, 3) array for RGB
    h, w = bin_pred.shape
    diff_mask = np.zeros((h, w, 3), dtype=np.uint8)

    tp = (bin_pred == 1) & (bin_tgt == 1)
    fp = (bin_pred == 1) & (bin_tgt == 0)
    fn = (bin_pred == 0) & (bin_tgt == 1)

    # Color coding
    # TP == green == (0,255,0)
    diff_mask[tp] = [0, 255, 0]
    # FP == yellow == (255,255,0)
    diff_mask[fp] = [255, 255, 0]
    # FN == red == (255,0,0)
    diff_mask[fn] = [255, 0, 0] # TN == remain black == (0,0,0)

    img_name = f"./med-sam/validation-diff-masks/diffmask_epoch{epoch}_batch{batch_idx}.png"
    save_path = os.path.join(save_dir, img_name)
    Image.fromarray(diff_mask).save(save_path)

# -------------------------------
# Validation Loop
# -------------------------------
def validate(model, dataloader, device, seg_loss_func, ce_loss_func, loss_type, epoch=0):
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, (images, masks, skeletons, bboxes) in enumerate(tqdm(dataloader, desc="Validating")):
            images = images.to(device)
            masks = masks.to(device)
            bboxes_np = bboxes.cpu().numpy()

            preds = model(images, bboxes_np)
            loss = compute_loss(preds, masks, seg_loss_func, ce_loss_func, loss_type)
            val_loss += loss.item()

            probs = torch.sigmoid(preds)
            bin_preds = (probs > 0.5).float()
            bin_masks = (masks > 0.5).float()
            iou = compute_iou(bin_preds, bin_masks.unsqueeze(1))
            val_iou += iou
            count += 1

            # Save the difference mask for the last batch in each epoch
            if batch_idx == len(dataloader):
                save_difference_mask(bin_preds, bin_masks.unsqueeze(1), epoch, batch_idx)

    return val_loss / count, val_iou / count

# -------------------------------
# Model Definition (MedSAM)
# -------------------------------
class MedSAM(nn.Module):
    def __init__(self, sam_model):
        super(MedSAM, self).__init__()
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder

        # Freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, images, box):
        image_embedding = self.image_encoder(images)
        box_torch = torch.as_tensor(box, dtype=torch.float32, device=images.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=box_torch, masks=None
        )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

# -------------------------------
# Main Training Routine
# -------------------------------
def train_medsam(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_message(logging.INFO, f"Using device: {device}")

    log_message(logging.INFO, "Preparing dataset... (same code as original script)")
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    skeleton_dir = args.skeleton_dir
    image_size = 1024  
    vess_dataset = VessMapDataset(image_dir, mask_dir, skeleton_dir, image_size, apply_transform=args.augment)

    train_loader, test_loader = vess_dataset.vess_map_dataloader(
        batch_size=args.batch_size, train_size=args.train_size / 100
    )

    class_weight_tensor = vess_dataset.class_weights_tensor
    #TODO: confirm if the weights are right
    ce_loss_func = nn.BCEWithLogitsLoss(pos_weight=class_weight_tensor[1]/class_weight_tensor[0], reduction="mean")
    seg_loss_func = monai.losses.DiceLoss(
        sigmoid=True, squared_pred=True, reduction="mean")
    

    log_message(logging.INFO, "Initializing Fabric...")
    fabric = L.Fabric(accelerator="cuda", devices=1, precision=args.precision)
    with fabric.init_module():
        log_message(logging.INFO, "Loading base SAM model checkpoint (ViT-B)...")
        MedSAM_CKPT_PATH = "../MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
        sam_model_inst = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
        medsam_model = MedSAM(sam_model_inst)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            [
                {'params': medsam_model.image_encoder.parameters()},
                {'params': medsam_model.mask_decoder.parameters()}
            ],
            lr=args.lr, momentum=0.9
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            [
                {'params': medsam_model.image_encoder.parameters()},
                {'params': medsam_model.mask_decoder.parameters()}
            ],
            lr=args.lr
        )
    else:
        raise ValueError("Unknown optimizer choice, choose from ['sgd','adam']")

    if args.scheduler == "cosine":
        num_steps = args.epochs * len(train_loader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    else:
        scheduler = None

    train_loader = fabric.setup_dataloaders(train_loader)
    test_loader = fabric.setup_dataloaders(test_loader)
    medsam_model, optimizer = fabric.setup(medsam_model, optimizer)
    device = fabric.device

    log_message(logging.INFO, "Training parameters:")
    for arg_name, val in vars(args).items():
        log_message(logging.INFO, f" - {arg_name}: {val}")

    date_str = datetime.datetime.now().strftime("%d%m%Y")
    model_name = f"medsam_vit_b_{args.optimizer}_lr{args.lr}_bs{args.batch_size}_{args.loss_type}_{args.train_size}pct_{date_str}"

    train_losses, val_losses, val_ious = [], [], []
    best_loss = float("inf")

    for epoch in range(args.epochs):
        log_message(logging.INFO, f"Epoch {epoch+1}/{args.epochs}")

        train_loss = train_one_epoch(
            model=medsam_model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            seg_loss_func=seg_loss_func,
            ce_loss_func=ce_loss_func,
            loss_type=args.loss_type,
            accumulate_grad_steps=args.accumulate_grad_steps,
            clip_grad=True,
            fabric=fabric
        )
        train_losses.append(train_loss)

        val_loss, val_iou = validate(
            model=medsam_model,
            dataloader=test_loader,
            device=device,
            seg_loss_func=seg_loss_func,
            ce_loss_func=ce_loss_func,
            loss_type=args.loss_type,
            epoch=epoch
        )
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        log_message(logging.INFO, f"Epoch {epoch+1}, TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, ValIoU={val_iou:.4f}")

        if scheduler is not None:
            scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            fabric.save(f"./med-sam/models/{model_name}_best.pth", {"model": medsam_model.state_dict()})
            log_message(logging.INFO, f"Best model saved at epoch {epoch+1} with val_loss {best_loss:.4f}")

        # Save latest
        fabric.save(f"./med-sam//models/{model_name}_latest.pth", {"model": medsam_model.state_dict()})

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_ious": val_ious
    }
    os.makedirs("./med-sam/metrics", exist_ok=True)
    with open(f"./med-sam/metrics/{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    log_message(logging.INFO, f"Metrics saved to ./med-sam/metrics/{model_name}_metrics.json")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    l1 = ax1.plot(range(1, args.epochs + 1), train_losses, label='Train Loss', color='tab:blue', linestyle='-')
    l2 = ax1.plot(range(1, args.epochs + 1), val_losses, label='Val Loss', color='tab:blue', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Val IoU', color='tab:red')
    l3 = ax2.plot(range(1, args.epochs + 1), val_ious, label='Val IoU', color='tab:red', linestyle='-')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')
    plt.title('Training & Validation Loss and IoU Over Epochs')
    plt.grid(True)
    os.makedirs("./med-sam/models", exist_ok=True)
    plt.savefig(f"./med-sam/models/{model_name}_training_validation_metrics.png")
    plt.close(fig)
    log_message(logging.INFO, f"Plot saved to ./med-sam/models/{model_name}_training_validation_metrics.png")

    os.makedirs("./med-sam/logs", exist_ok=True)
    with open(f"./med-sam/logs/{model_name}_logs.json", "w") as f:
        json.dump(LOG_RECORDS, f, indent=4)
    log_message(logging.INFO, f"Logs saved to ./med-sam/logs/{model_name}_logs.json")

def get_args():
    parser = argparse.ArgumentParser(
        description="Fine tune a MEDSAM on the VessMapDataset."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd","adam"], help="Optimizer choice")
    parser.add_argument("--loss_type", type=str, default="both", choices=["dice","ce","both"], help="Loss type")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine","none"], help="Scheduler type")
    parser.add_argument("--train_size", type=float, default=80, help="Train/validation split ratio as a percentage")
    parser.add_argument("--accumulate_grad_steps", type=int, default=1, help="Accumulate grad steps")
    parser.add_argument("--image_dir", type=str, default="../data/vess-map/images", help="Images directory")
    parser.add_argument("--mask_dir", type=str, default="../data/vess-map/labels", help="Labels directory")
    parser.add_argument("--skeleton_dir", type=str, default="../data/vess-map/skeletons", help="Skeleton directory")
    parser.add_argument("--precision", type=str, default="bf16-mixed", choices=["bf16-mixed","32-true"], help="Fabric precision")
    parser.add_argument("--augment", type=bool, default=True, help="Apply random data augmentation")
    return parser.parse_args()

if __name__ == "__main__":
    logger = logging.getLogger("MedSAM_Training")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    args = get_args()
    train_medsam(args)

    log_message(logging.INFO, "Training completed.")
