import os
import numpy as np
from glob import glob
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    AsChannelFirstd,
    Compose,
    LoadImaged,
    RandFlipd,
    RandSpatialCropd,
    Resized,
    ScaleIntensityRanged,
    ToTensord,
)
import torch
import ubelt as ub
import pint
import site
from util.ARGUS_Transforms import ARGUS_RandSpatialCropSlicesd
Ureg = pint.UnitRegistry()

def setup_vfold_files(img_dir, p_prefix, num_folds):
    all_train_images = [sorted(glob(os.path.join(img_dir,"ONUS-"+ x + "HV","*.mha"))) for x in p_prefix]
    all_train_images = [i for img in all_train_images for i in img]
    total_bytes = 0
    for p in all_train_images:
        p = ub.Path(p)
        total_bytes += p.stat().st_size
    print("\n")
    print("Total size of images in the dataset: ")
    print((total_bytes * Ureg.byte).to("GiB"))

    num_images = len(all_train_images)
    
    print("\n")
    print("Num images = ", num_images)
    print("\n")

    train_files = []
    for i in range(len(all_train_images)):
        train_files.append(
            [
                {"image": all_train_images[i]}
            ]
        )
    print(
        len(train_files)
    )
    return train_files

def setup_training_vfold(train_files, num_slices):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AsChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                a_min=0, a_max=255,
                b_min=0.0, b_max=1.0,
                keys=["image"]),
            ARGUS_RandSpatialCropSlicesd(
                num_slices=num_slices,
                axis=0,
                reduce_to_statistics=True,
                extended=False,
                include_center_slice=True,
                include_gradient=True,
                keys=["image"]),
            RandFlipd(prob=0.5, 
                spatial_axis=0,
                keys=["image"]),
            ToTensord(dtype=torch.float, keys=["image"])
        ]
    )

    train_ds = Dataset(
        data=train_files,
        transform=train_transforms,
    )
    return train_ds