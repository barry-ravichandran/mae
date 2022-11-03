import os
import numpy as np
from glob import glob
from monai.data import CacheDataset, DataLoader
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
site.addsitedir('/data/barry.ravichandran/repos/AnatomicRecon-POCUS-AI/ARGUS')
from ARGUS_Transforms import ARGUS_RandSpatialCropSlicesd
Ureg = pint.UnitRegistry()

def setup_vfold_files(img_dir, p_prefix, num_folds):
    all_train_images = [sorted(glob(os.path.join(img_dir,"ONUS-"+ x + "HV", "butterfly-iq", "*.mha"))) for x in p_prefix]
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

    fold_prefix_list = []
    p_count = 0
    for i in range(num_folds):
        num_p = 1
        f = []
        if p_count < len(p_prefix):
            for p in range(num_p):
                f.append([p_prefix[p_count + p]])
        p_count += num_p
        fold_prefix_list.append(f)

    for i in range(num_folds):
        print(i, fold_prefix_list[i])
    print("\n")

    train_files = []

    for i in range(num_folds):
        tr_folds = []
        for f in range(i, i + num_folds):
            tr_folds.append(fold_prefix_list[f % num_folds])
        tr_folds = list(np.concatenate(tr_folds).flat)
        train_files.append(
            [
                {"image": img}
                for img in zip(
                    [
                        im
                        for im in all_train_images
                        if any(pref in im for pref in tr_folds)
                    ]
                )
            ]
        )
        print(
            len(train_files[i])
        )
        return train_files

def setup_training_vfold(train_files, num_slices, vfold_num):
    train_transforms = Compose(
        [
        LoadImaged(keys=["image"]),
        AsChannelFirstd(keys='image'),
        ScaleIntensityRanged(
            a_min=0, a_max=255,
            b_min=0.0, b_max=1.0,
            keys=["image"]),
        ARGUS_RandSpatialCropSlicesd(
            num_slices=[num_slices],
            axis=0,
            reduce_to_statistics=[True],
            require_labeled=True,
            extended=False,
            include_center_slice=True,
            include_gradient=True,
            keys=['image']),
        RandFlipd(prob=0.5, 
            spatial_axis=0,
            keys=['image']),
        ToTensord(keys=["image"], dtype=torch.float)
        ])

    cache_rate_train = 1.0
    num_workers_train = 8
    batch_size_train = 4

    train_ds = CacheDataset(
        data=train_files[vfold_num],
        transform=train_transforms,
        cache_rate=cache_rate_train,
        num_workers=num_workers_train,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers_train,
    )

    return train_loader