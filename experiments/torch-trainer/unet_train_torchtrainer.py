import torch
import segmentation_models_pytorch as smp

# Import necessary components from torchtrainer
from torchtrainer.train import DefaultTrainer
from torchtrainer.util.train_util import (
    seed_all,
    Logger,
    LoggerPlotter,
    WrapDict,
)
from torchtrainer.metrics.confusion_metrics import ConfusionMatrixMetrics


class UnetVessTrainer(DefaultTrainer):
    """
    A custom trainer for UNet segmentation on VessMAP.

    Expects:
      - dataset_class to be "vessmap"
      - model_class to be "unet_smp"

    It uses get_dataset_vessmap_train to load the training and validation splits
    and instantiates a segmentation_models_pytorch UNet (ResNet-50 encoder).
    Also, it overrides the dataset setup to convert BCE pos_weight to a tensor.
    """

    def get_dataset(
        self,
        dataset_class,
        dataset_path,
        split_strategy,
        resize_size,
        augmentation_strategy,  # TODO: mudar quando for usar
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

            ds_train, ds_valid, *dataset_props =  get_dataset_vessmap_few_train(
                dataset_path,
                split_strategy,
                resize_size=resize_size,
            )
            class_weights, ignore_index, collate_fn = dataset_props

            return ds_train, ds_valid, class_weights, ignore_index, collate_fn

    def get_model(
        self, model_class, weights_strategy, num_classes, num_channels, **model_params
    ):
        if model_class == "unet_smp" or model_class == "Unet":
            model = smp.Unet(
                encoder_name="resnet50",
                encoder_weights=(
                    "imagenet" if weights_strategy is None else weights_strategy
                ),
                in_channels=num_channels,
                classes=num_classes,  # For binary segmentation, typically set to 1.
            )
            return model
        else:
            raise NotImplementedError(
                f"Model class '{model_class}' not supported. Use 'unet_smp'."
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

        # Load the dataset.
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
            # Create the standard BCEWithLogitsLoss with pos_weight as a tensor.
            bce_loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    class_weights[1] / class_weights[0], device=args.device
                ),
                reduction="mean",
            )

            # Wrap the BCE loss to squeeze the channel dimension of the model output.
            def loss_func(input, target):
                # input shape: [B, 1, H, W]; target shape: [B, H, W]
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

        # For BCE, we force the number of classes to 1.
        num_classes = 1
        # Infer number of channels from the first training sample.
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


# Main entry point: instantiate and run the trainer.
if __name__ == "__main__":
    trainer = UnetVessTrainer()
    trainer.fit()
# TODO: check parameters

"""
unet_train_torchtrainer.py

An example training script using the torchtrainer module to train a UNet
(from segmentation_models_pytorch) on the VessMAP dataset.

Usage (example):
    python \
        /home/fonta42/Desktop/masters-degree/experiments/torch-trainer/unet_train_torchtrainer.py \
        /home/fonta42/Desktop/masters-degree/data/torch-trainer/VessMAP \
        vessmap \
        Unet \
        --experiments_path /home/fonta42/Desktop/masters-degree/experiments/torch-trainer/unet_runs/ \
        --experiment_name train_unet \
        --run_name unet_rand_0.2_none_256x256_bce_10_0.001_1_1 \
        --validate_every 50 \
        --val_img_indices 0 1 2 3 \
        --split_strategy rand_0.2 \
        --augmentation_strategy None \
        --resize_size 256 256 \
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
