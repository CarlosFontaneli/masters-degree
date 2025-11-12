import os
import sys
import json
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import monai
from torch.nn.utils import clip_grad_norm_
from tqdm.notebook import tqdm
from pathlib import Path

sys.path.append("../../data/vess-map")
from vess_map_dataset import VessMapDataset

sys.path.append("/home/fonta42/Desktop/masters-degree/torchtrainer/torchtrainer")
from torchtrainer.models.litemedsam.tiny_vit_sam import TinyViT
from torchtrainer.models.medsam.segment_anything.modeling import (
    MaskDecoder,
    PromptEncoder,
    TwoWayTransformer,
)
from torchtrainer.models.litemedsam.litemedsam import LiteMedSAM


LOG_RECORDS = []


def log_message(level, message):
    logger = logging.getLogger("Experiment_Runner")
    logger.log(level, message)
    LOG_RECORDS.append(
        {
            "level": logging.getLevelName(level),
            "message": message,
            "time": datetime.datetime.now().isoformat(),
        }
    )


def create_difference_mask(bin_preds, bin_targets):
    b, _, h, w = bin_preds.shape
    diff_masks = np.zeros((b, h, w, 3), dtype=np.uint8)
    for idx in range(b):
        bin_pred = bin_preds[idx, 0]
        bin_tgt = bin_targets[idx, 0]
        tp = (bin_pred == 1) & (bin_tgt == 1)  # True Positive
        fp = (bin_pred == 1) & (bin_tgt == 0)  # False Positive
        fn = (bin_pred == 0) & (bin_tgt == 1)  # False Negative
        diff_masks[idx][tp] = [0, 255, 0]  # Green
        diff_masks[idx][fp] = [255, 255, 0]  # Yellow
        diff_masks[idx][fn] = [255, 0, 0]  # Red
    return diff_masks


def save_comparison_plot(
    original_img, bin_pred, diff_mask, epoch, batch_idx, model_name
):
    if torch.is_tensor(original_img):
        original_img = original_img.cpu().numpy()
    if torch.is_tensor(bin_pred):
        bin_pred = bin_pred.cpu().numpy()

    original_img = np.transpose(original_img, (1, 2, 0))
    output_dir = f"./smallers_lite_medsam/lite_medsam/validation-diff-masks/{model_name}_validation-diff-masks"
    os.makedirs(output_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_img, cmap="gray")
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[1].imshow(bin_pred, cmap="gray")
    axs[1].set_title("Prediction")
    axs[1].axis("off")
    axs[2].imshow(diff_mask)
    axs[2].set_title("Diff Mask (TP:Green, FP:Yellow, FN:Red)")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"comparison_epoch{epoch}_batch{batch_idx}.png")
    )
    plt.close(fig)


def compute_loss(preds, masks, seg_loss_func, ce_loss_func, loss_type):
    """Computes loss for single (UNet) or dual (WNet) outputs."""

    def _calculate_loss(p, m):
        if loss_type == "dice":
            return seg_loss_func(p, m)
        elif loss_type == "ce":
            return ce_loss_func(p, m.float())
        elif loss_type == "both":
            return seg_loss_func(p, m) + ce_loss_func(p, m.float())
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    if isinstance(preds, tuple):  # Handle WNet's dual output
        loss1 = _calculate_loss(preds[0], masks)
        loss2 = _calculate_loss(preds[1], masks)
        return loss1 + loss2
    else:  # Handle UNet's single output
        return _calculate_loss(preds, masks)


def train_one_epoch(
    model,
    dataloader,
    device,
    optimizer,
    seg_loss_func,
    ce_loss_func,
    loss_type,
    accumulate_grad_steps=1,
    clip_grad=True,
):
    model.train()
    running_loss = 0.0
    running_dice = 0.0  # <-- Add this to track dice

    for step, (images, masks, _) in enumerate(
        tqdm(dataloader, desc="Training", leave=False)
    ):
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = compute_loss(preds, masks, seg_loss_func, ce_loss_func, loss_type)

        loss = loss / accumulate_grad_steps
        loss.backward()

        if (step + 1) % accumulate_grad_steps == 0:
            if clip_grad:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulate_grad_steps

        with torch.no_grad():
            dice_loss = seg_loss_func(
                preds,
                masks,
            )

            running_dice += dice_loss.item()
    # Return both average loss and average dice for the epoch
    return running_loss / len(dataloader), running_dice / len(dataloader)


def validate(
    model,
    dataloader,
    device,
    seg_loss_func,
    ce_loss_func,
    loss_type,
    epoch=0,
    model_name="default_model",
):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for batch_idx, (images, masks, _) in enumerate(
            tqdm(dataloader, desc="Validating", leave=False)
        ):
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = compute_loss(preds, masks, seg_loss_func, ce_loss_func, loss_type)
            val_loss += loss.item()

            # For metrics and visualization, use the final output
            final_preds = preds

            dice_loss = seg_loss_func(
                preds,
                masks,
            )
            val_dice += dice_loss.item()

            # Save a visualization plot periodically from the last batch
            if batch_idx == len(dataloader) - 1 and epoch % 10 == 0 and epoch > 0:
                probs = torch.sigmoid(final_preds)
                bin_preds = (probs > 0.5).float()
                bin_masks = (masks > 0.5).float()
                diff_masks = create_difference_mask(
                    bin_preds.cpu().numpy(), bin_masks.cpu().numpy()
                )
                save_comparison_plot(
                    original_img=images[0],
                    bin_pred=bin_preds[0, 0],
                    diff_mask=diff_masks[0],
                    epoch=epoch,
                    batch_idx=batch_idx,
                    model_name=model_name,
                )

    return val_loss / len(dataloader), val_dice / len(dataloader)


def train_model(args, model, model_name_prefix, device):
    log_message(
        logging.INFO,
        f"Starting training for model: {model_name_prefix} with {args.train_size}% data",
    )

    vess_dataset = VessMapDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        skeleton_dir=args.skeleton_dir,
        image_size=args.image_size,
        apply_transform=args.augment,
    )
    train_loader, test_loader = vess_dataset.vess_map_dataloader(
        batch_size=args.batch_size, train_size=args.train_size / 100
    )

    model.to(device)

    class_weight_tensor = vess_dataset.class_weights_tensor.to(device)
    ce_loss_func = nn.BCEWithLogitsLoss(
        pos_weight=class_weight_tensor[1] / class_weight_tensor[0], reduction="mean"
    )
    seg_loss_func = monai.losses.DiceLoss(
        sigmoid=True, squared_pred=True, reduction="mean"
    )

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unknown optimizer choice. Choose from ['sgd', 'adam'].")

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * len(train_loader)
        )

    date_str = datetime.datetime.now().strftime("%d%m%Y")
    # Add train_size to model name for unique saving
    model_name = f"{model_name_prefix}_{args.train_size}pct_{args.optimizer}_lr{args.lr}_bs{args.batch_size}_{date_str}"

    train_losses, val_losses, train_dices, val_dices = [], [], [], []
    best_loss = float("inf")

    # Main training loop
    for epoch in range(args.epochs):
        train_loss, train_dice = train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            seg_loss_func,
            ce_loss_func,
            args.loss_type,
            args.accumulate_grad_steps,
        )
        train_losses.append(train_loss)
        train_dices.append(train_dice)

        val_loss, val_dice = validate(
            model,
            test_loader,
            device,
            seg_loss_func,
            ce_loss_func,
            args.loss_type,
            epoch,
            model_name,
        )
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        log_message(
            logging.INFO,
            f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}",
        )

        if scheduler:
            scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs("./smallers_lite_medsam/lite_medsam/models", exist_ok=True)
            torch.save(
                model.state_dict(),
                f"./smallers_lite_medsam/lite_medsam/models/{model_name}_best.pth",
            )
            log_message(
                logging.INFO,
                f"Best model saved at epoch {epoch+1} with val_loss {best_loss:.4f}",
            )

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_dices": train_dices,
        "val_dices": val_dices,
    }

    os.makedirs("./smallers_lite_medsam/lite_medsam/metrics", exist_ok=True)
    with open(
        f"./smallers_lite_medsam/lite_medsam/metrics/{model_name}_metrics.json", "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    return metrics


class TrainingConfig:
    # --- Training Hyperparameters ---
    lr = 1e-4
    batch_size = 1
    epochs = 5
    optimizer = "adam"  # 'sgd' or 'adam'
    momentum = 0.9
    loss_type = "dice"  # 'dice', 'ce', or 'both'
    scheduler = "none"  # 'cosine' or 'none'
    accumulate_grad_steps = 1

    # --- Dataset & Augmentation ---
    train_size = 80  # 80% for training, 20% for validation
    image_size = 256
    augment = True

    image_dir = "../../data/vess-map/images"
    mask_dir = "../../data/vess-map/labels"
    skeleton_dir = "../../data/vess-map/skeletons"


def create_litemedsam_model(model_params, freeze_image_encoder, load_checkpoint=True):
    """
    A robust factory function to instantiate a new LiteMedSAM model
    from a dictionary of parameters.
    Initializes everything on CPU first, then allows moving the final model.
    Allows loading from checkpoint (default) or training from scratch.
    """
    # Define the device as CPU for initialization
    init_device = "cpu"

    # 1. Extract parameters from the dictionary
    tiny_vit_params = model_params["tiny_vit"]
    prompt_encoder_params = model_params["prompt_encoder"]
    mask_decoder_params = model_params["mask_decoder"]

    # Extract img_size for the LiteMedSAM wrapper class
    img_size = tiny_vit_params["img_size"]

    # 2. Instantiate NEW components on the CPU
    image_encoder = TinyViT(**tiny_vit_params).to(init_device)

    prompt_encoder = PromptEncoder(**prompt_encoder_params).to(init_device)

    # Handle nested transformer parameters for MaskDecoder
    transformer = TwoWayTransformer(**mask_decoder_params["transformer_params"])

    # Create a copy of mask_decoder_params and remove the nested dict
    # to avoid "unexpected keyword argument" error.
    decoder_args = mask_decoder_params.copy()
    del decoder_args["transformer_params"]

    mask_decoder = MaskDecoder(transformer=transformer, **decoder_args).to(init_device)

    # 3. Call LiteMedSAM.__init__ (all components and internal bbox are on CPU)
    model = LiteMedSAM(
        image_encoder=image_encoder,
        mask_decoder=mask_decoder,
        prompt_encoder=prompt_encoder,
        img_size=(img_size, img_size),  # Use img_size from params
        freeze_image_encoder=freeze_image_encoder,
    )

    # 4. Load the checkpoint IF requested (to CPU)
    if load_checkpoint:
        try:
            base_path = "/home/carlos/Desktop/masters-degree/torchtrainer/torchtrainer"
            script_directory = Path(f"{base_path}/models/litemedsam/").resolve()
            ckpt_path = script_directory / "lite_medsam.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found. Searched: {ckpt_path}")

            # Load checkpoint to CPU
            medsam_lite_ckpt = torch.load(
                ckpt_path, map_location=init_device, weights_only=True
            )
            model.load_state_dict(medsam_lite_ckpt, strict=False)
            print(f"Checkpoint loaded successfully from {ckpt_path} to CPU")

        except Exception as e:
            print(
                f"WARNING: Could not load pretrained checkpoint: {e}. Model initialized with random weights on CPU."
            )
    else:
        print(
            "Checkpoint loading skipped. Model initialized with random weights on CPU."
        )

    # 5. Return the model (still on CPU)
    return model


def main():
    logger = logging.getLogger("Experiment_Runner")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_percentages = [80]

    class ExperimentConfig(TrainingConfig):
        epochs = 2

    config_path = "./small_vit_model_definitions.json"  
    all_model_configs = {}
    try:
        with open(config_path, 'r') as f:
            all_model_configs = json.load(f)
        log_message(logging.INFO, f"Successfully loaded {len(all_model_configs)} model configs from {config_path}")
    except Exception as e:
        log_message(logging.ERROR, f"Failed to load {config_path}: {e}")
        return  # Stop execution if configs can't be loaded

    models_to_run = all_model_configs

    all_results = {}

    for model_key, model_params in models_to_run.items():
        freeze_image_encoder = False
        load_checkpoint = False

        log_message(logging.INFO, f"========================================")
        log_message(logging.INFO, f"Starting experiments for model: {model_key}")
        log_message(logging.INFO, f"========================================")

        all_results[model_key] = {}

        for percent in train_percentages:
            args = ExperimentConfig()
            args.train_size = percent

            log_message(
                logging.INFO,
                f"Instantiating model {model_key} for {percent}% data on CPU",
            )
            model = create_litemedsam_model(
                model_params=model_params,
                freeze_image_encoder=freeze_image_encoder,
                load_checkpoint=load_checkpoint,
            )

            # 2. Move the fully constructed model to the target device (GPU/CPU)
            log_message(logging.INFO, f"Moving model {model_key} to {device}")
            model.to(device)

            # 3. Run training and get metrics
            metrics = train_model(args, model, model_key, device)

            all_results[model_key][percent] = {
                "train_losses": metrics["train_losses"],
                "val_losses": metrics["val_losses"],
                "train_dices": metrics["train_dices"],
                "val_dices": metrics["val_dices"],
                "max_val_dice": (
                    max(metrics["val_dices"]) if metrics["val_dices"] else 0
                ),
                "max_train_dice": (
                    max(metrics["train_dices"]) if metrics["train_dices"] else 0
                ),
                "min_val_loss": (
                    min(metrics["val_losses"])
                    if metrics["val_losses"]
                    else float("inf")
                ),
                "min_train_loss": (
                    min(metrics["train_losses"])
                    if metrics["train_losses"]
                    else float("inf")
                ),
            }
            log_message(
                logging.INFO,
                f"Finished run for {model_key} with {percent}% data. Max Val Dice: {all_results[model_key][percent]['max_val_dice']:.4f}",
            )

    log_message(logging.INFO, "🎉 All experiments completed successfully!")
    return all_results


experiment_results = main()

os.makedirs("./smallers_lite_medsam/lite_medsam/results", exist_ok=True)
with open(
    "./smallers_lite_medsam/lite_medsam/results/all_experiment_results.json", "w"
) as f:
    json.dump(experiment_results, f, indent=4)
