import os
import sys
import torch
import monai
import torch.nn as nn
import torch.optim as optim
import argparse

# Import necessary components from torchtrainer
from torchtrainer.train import DefaultTrainer, DefaultModuleRunner
from torchtrainer.util.train_util import (
    seed_all,
    Logger,
    LoggerPlotter,
    ParseKwargs,
    ParseText,
    WrapDict,
    to_csv_nan,
)
from torchtrainer.metrics.confusion_metrics import ConfusionMatrixMetrics

from torchtrainer.datasets.vessel import get_dataset_vessmap_train


# Append MedSAM directory and import SAM model registry
sys.path.append("/home/fonta42/Desktop/masters-degree/MedSAM")
from segment_anything import sam_model_registry


# -------------------------------
# MedSAM Model Definition
# -------------------------------
class MedSAM(nn.Module):
    def __init__(self, sam_model):
        super(MedSAM, self).__init__()
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder
        # Freeze all prompt encoder parameters.
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
        ori_res_masks = nn.functional.interpolate(
            low_res_masks,
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


# -------------------------------
# MedSAM Adapter Class
# -------------------------------
class MedSAMAdapter(MedSAM):
    """
    A new adapter class inheriting from MedSAM.
    This adapter overrides the forward method so that if no bounding box is provided,
    a bounding box covering the full image is generated automatically.
    Additionally, if the input images are grayscale (1 channel), they are converted to RGB (3 channels)
    by replicating the single channel.
    This makes the model compatible with datasets that may supply grayscale images.
    """

    def forward(self, images, box=None):
        # Check if images have 1 channel and convert to 3 channels if needed.
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # If no bounding box is provided, generate a full-image bbox.
        if box is None:
            B, _, H, W = images.shape
            # Create a bbox covering the full image and repeat it for each sample in the batch.
            box = torch.tensor([0, 0, W, H], dtype=torch.float32, device=images.device)
            box = box.unsqueeze(0).repeat(B, 1)  # Now the shape is [B, 4]

        # Call the parent forward method.
        return super().forward(images, box)


# -------------------------------
# Custom Module Runner for MedSAM
# -------------------------------
class MedSAMModuleRunner(DefaultModuleRunner):
    """
    A custom module runner that calls the model with both images and (if available) bounding boxes.
    In this case, the MedSAMAdapter automatically generates the bbox if not provided.
    """

    def train_one_epoch(self, epoch: int):
        self.model.train()
        scaler = self.scaler
        profiler = self.profiler
        dl_iter = iter(self.dl_train)
        pbar = torch.tqdm(
            range(len(self.dl_train)),
            desc="Training",
            leave=False,
            unit="batchs",
            dynamic_ncols=True,
            colour="blue",
            disable=self.args.disable_tqdm,
        )
        if epoch == 1:
            profiler.start("train")
        for batch_idx in pbar:
            with profiler.section(f"data_{batch_idx}"):
                # Standard VessMap dataset returns images and masks (no bbox).
                imgs, targets = next(dl_iter)
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            with profiler.section(f"forward_{batch_idx}"):
                self.optim.zero_grad()
                with torch.autocast(
                    device_type=self.device, enabled=scaler.is_enabled()
                ):
                    # The adapter will generate a full-image bounding box.
                    scores = self.model(imgs)
                    loss = self.loss_func(scores, targets)
            with profiler.section(f"backward_{batch_idx}"):
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
            profiler.step()
            self.logger.log(
                epoch, batch_idx, "Train loss", loss.detach(), imgs.shape[0]
            )
        self.logger.log_epoch(epoch, "lr", self.optim.param_groups[0]["lr"])
        self.scheduler.step()

    @torch.no_grad()
    def validate_one_epoch(self, epoch: int):
        self.model.eval()
        profiler = self.profiler
        dl_iter = iter(self.dl_valid)
        pbar = torch.tqdm(
            range(len(self.dl_valid)),
            desc="Validating",
            leave=False,
            unit="batchs",
            dynamic_ncols=True,
            colour="green",
            disable=self.args.disable_tqdm,
        )
        if epoch == 1:
            profiler.start("validation")
        for batch_idx in pbar:
            with profiler.section(f"data_{batch_idx}"):
                imgs, targets = next(dl_iter)
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            with profiler.section(f"forward_{batch_idx}"):
                scores = self.model(imgs)
                loss = self.loss_func(scores, targets)
            with profiler.section(f"metrics_{batch_idx}"):
                self.logger.log(
                    epoch, batch_idx, "Validation loss", loss, imgs.shape[0]
                )
                for perf_func in self.perf_funcs:
                    results = perf_func(scores, targets)
                    for name, value in results.items():
                        self.logger.log(epoch, batch_idx, name, value, imgs.shape[0])
            profiler.step()


# -------------------------------
# Custom Trainer for MedSAM Model
# -------------------------------
class MedsamTrainer(DefaultTrainer):
    """
    A custom trainer for MedSAM segmentation on VessMAP.

    Expects:
      - dataset_class to be "medsam"
      - model_class to be "medsam"

    Adaptations include:
      - Casting BCE pos_weight to a tensor.
      - Overriding training/validation loops to work without an explicit bbox (adapter generates it).
      - Casting target tensors to float in the loss wrapper.
      - Using additional parameters (e.g. image_size and augment) passed via --dataset_params.
    """

    def __init__(self, param_dict: dict | None = None):
        # Override the module runner with the custom MedSAMModuleRunner.
        self.module_runner = MedSAMModuleRunner()
        super().__init__(param_dict)

    def get_dataset(
        self,
        dataset_class,
        dataset_path,
        split_strategy,
        resize_size,
        augmentation_strategy,
        **dataset_params,
    ):
        if dataset_class == "vessmap":
            from torchtrainer.datasets.vessel import get_dataset_vessmap_train

            ds_train, ds_valid, class_weights, ignore_index, collate_fn = (
                get_dataset_vessmap_train(
                    dataset_path,
                    split_strategy=split_strategy,
                    resize_size=resize_size,  # TODO: mudar quando for usar
                )
            )
            return ds_train, ds_valid, class_weights, ignore_index, collate_fn
        if dataset_class == "vessmap_few":
            from vessmap_few_dataset import get_dataset_vessmap_few_train

            ds_train, ds_valid, *dataset_props = get_dataset_vessmap_few_train(
                dataset_path,
                split_strategy,
                resize_size=resize_size,
            )
            class_weights, ignore_index, collate_fn = dataset_props

            return ds_train, ds_valid, class_weights, ignore_index, collate_fn

    def get_model(
        self, model_class, weights_strategy, num_classes, num_channels, **model_params
    ):
        if model_class.lower() == "medsam":
            checkpoint_path = (
                weights_strategy
                if weights_strategy is not None
                else "../../MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
            )
            sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            # Use the adapter so that the model can generate bounding boxes automatically.
            model = MedSAMAdapter(sam_model)
            return model
        else:
            raise NotImplementedError(
                f"Model class '{model_class}' not supported. Use 'medsam'."
            )

    def setup_dataset(self):
        """Override setup_dataset to update the BCE loss pos_weight and adjust input shapes."""
        args = self.args
        dataset_class = args.dataset_class
        dataset_path = args.dataset_path
        split_strategy = args.split_strategy
        resize_size = args.resize_size
        augmentation_strategy = args.augmentation_strategy
        dataset_params = args.dataset_params

        seed_all(args.seed)

        ds_train, ds_valid, class_weights, ignore_index, collate_fn = self.get_dataset(
            dataset_class,
            dataset_path,
            split_strategy,
            resize_size,
            augmentation_strategy,
            **dataset_params,
        )

        if args.ignore_class_weights:
            class_weights = (1.0,) * len(class_weights)
        if ignore_index is None:
            ignore_index = -100

        loss_function = args.loss_function
        if loss_function == "cross_entropy":
            loss_func = torch.nn.CrossEntropyLoss(
                torch.tensor(class_weights, device=args.device),
                ignore_index=ignore_index,
            )
        elif loss_function == "single_channel_cross_entropy":
            from torchtrainer.metrics.losses import SingleChannelCrossEntropyLoss

            loss_func = SingleChannelCrossEntropyLoss(
                torch.tensor(class_weights, device=args.device),
                ignore_index=ignore_index,
            )
        elif loss_function == "bce":
            if ignore_index != -100:
                raise ValueError("The BCE loss does not support ignore_index")
            bce_loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    class_weights[1] / class_weights[0], device=args.device
                ),
                reduction="mean",
            )

            def loss_func(input, target):
                # Expected input shape: [B, 1, H, W]; target shape: [B, H, W]
                return bce_loss(input.squeeze(1), target.float())

        else:
            raise ValueError(f"Loss function {loss_function} not recognized")

        conf_metrics = ConfusionMatrixMetrics(ignore_index=ignore_index)
        perf_funcs = [
            WrapDict(conf_metrics, ["Accuracy", "IoU", "Precision", "Recall", "Dice"])
        ]
        logger = Logger()
        logger_plotter = LoggerPlotter(
            [
                {"names": ["Train loss", "Validation loss"], "y_max": 1.0},
                {
                    "names": ["Accuracy", "IoU", "Precision", "Recall", "Dice"],
                    "y_max": 1.0,
                },
            ]
        )

        # For BCE, force number of classes to 1.
        num_classes = 1
        num_channels = ds_train[0][0].shape[0]
        self.module_runner.add_dataset_elements(
            ds_train,
            ds_valid,
            num_classes,
            num_channels,
            collate_fn,
            loss_func,
            perf_funcs,
            logger,
            logger_plotter,
        )


if __name__ == "__main__":
    trainer = MedsamTrainer()
    trainer.fit()
# TODO: check parameters

"""
medsam_train_torchtrainer.py

An example training script using the torchtrainer module to train a MedSAM model
(using the MedSAM model definition from MedSAM) on the VessMAP dataset.

Usage (example):
    python \
        /home/fonta42/Desktop/masters-degree/experiments/torch-trainer/medsam_train_torchtrainer.py \
        /home/fonta42/Desktop/masters-degree/data/torch-trainer/VessMAP \
        vessmap \
        MedSam \
        --experiments_path /home/fonta42/Desktop/masters-degree/experiments/torch-trainer/medsam_runs/ \
        --experiment_name train_medsam \
        --run_name medsam_rand_0.2_none_1024x1024_bce_10_0.001_1_1 \
        --validate_every 50 \
        --val_img_indices 0 1 2 3 \
        --split_strategy rand_0.2 \
        --augmentation_strategy None \
        --resize_size 1024 1024 \
        --loss_function bce \
        --num_epochs 10 \
        --validation_metric Dice \
        --lr 0.001 \
        --lr_decay 1.0 \
        --bs_train 1 \
        --bs_valid 1 \
        --weight_decay 0.0 \
        --optimizer adam \
        --momentum 0.9 \
        --seed 42 \
        --device cuda:0 \
        --num_workers 5
"""
