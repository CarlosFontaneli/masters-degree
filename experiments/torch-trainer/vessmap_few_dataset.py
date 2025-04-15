from pathlib import Path
import torch
from torchtrainer.datasets.vessel_base import VessMAP
from torchvision import tv_tensors
from torchvision.transforms import v2 as tv_transf
from torchvision.transforms.v2 import functional as tv_transf_F


class TrainTransforms:
    """Lightweight training transform for vessel datasets."""

    def __init__(self, resize_size, resize_target = True):

        self.resize_size = resize_size
        self.resize_target = resize_target

        scale = tv_transf.RandomAffine(degrees=0, scale=(0.95, 1.20))
        transl = tv_transf.RandomAffine(degrees=0, translate=(0.05, 0))
        rotate = tv_transf.RandomRotation(degrees=45)
        scale_transl_rot = tv_transf.RandomChoice((scale, transl, rotate))
        #brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
        #jitter = tv_transf.ColorJitter(brightness, contrast, saturation, hue)
        hflip = tv_transf.RandomHorizontalFlip()
        vflip = tv_transf.RandomVerticalFlip()

        self.transform = tv_transf.Compose((
            scale_transl_rot,
            #jitter,
            hflip,
            vflip,
        ))

    def __call__(self, img, target):

        img = torch.from_numpy(img).permute(2, 0, 1).to(dtype=torch.uint8)
        target = torch.from_numpy(target).unsqueeze(0).to(dtype=torch.uint8)

        img = tv_transf_F.resize(img, self.resize_size)
        if self.resize_target:
            target = tv_transf_F.resize(target, 
                                        self.resize_size, 
                                        interpolation=tv_transf.InterpolationMode.NEAREST_EXACT)

        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)

        img, target = self.transform(img, target)

        img = img.data.float()/255
        target = target.data.to(dtype=torch.int64)[0]

        return img, target

class ValidTransforms:
    """Validation transform that only resizes the image."""

    def __init__(self, resize_size = None, resize_target = True):
        self.resize_size = resize_size
        self.resize_target = resize_target

    def __call__(self, img, target):

        img = torch.from_numpy(img).permute(2, 0, 1).to(dtype=torch.uint8)
        target = torch.from_numpy(target).unsqueeze(0).to(dtype=torch.uint8)
        
        if self.resize_size is not None:
            img = tv_transf_F.resize(img, self.resize_size)
            if self.resize_target:
                target = tv_transf_F.resize(target, 
                                            self.resize_size, 
                                            interpolation=tv_transf.InterpolationMode.NEAREST_EXACT)

        img = img.float()/255
        target = target.to(dtype=torch.int64)[0]

        return img, target


def get_dataset_vessmap_few_train(
        dataset_path, 
        split_strategy, 
        resize_size=(256, 256), 
        ):
    """Get the VessMAP dataset for training.

    Parameters
    ----------
    dataset_path
        Path to the dataset root folder
    split_strategy
        Comma-separated list of samples to use for training
    resize_size
        Size to resize the images
    """

    class_weights = (0.26, 0.74)
    ignore_index = None
    collate_fn = None

    dataset_path = Path(dataset_path)

    names_train = split_strategy.split(",")

    ds = VessMAP(dataset_path, keepdim=True)

    train_images = []
    valid_images = []
    for image in ds.images:
        found = False
        for name in names_train:
            if name==image.stem:
                found = True
            
        if found:
            train_images.append(image.name)
        else:
            valid_images.append(image.name)   
    

    ds_train = VessMAP(dataset_path, keepdim=True, files=train_images)
    ds_valid = VessMAP(dataset_path, keepdim=True, files=valid_images)
        
    ds_train.transforms = TrainTransforms(resize_size)
    ds_valid.transforms = ValidTransforms(resize_size)

    return ds_train, ds_valid, class_weights, ignore_index, collate_fn

