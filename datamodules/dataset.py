"""
This module contains the definition of the VOC2012SegmentationDataset class.
"""
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class VOC2012SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        split="train",
        image_transforms=None,
        mask_transforms=None,
        train_file="",
        val_file="",
    ):
        self.split = split
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.image_dir = os.path.join(image_dir)
        self.mask_dir = os.path.join(mask_dir)
        self.image_files = os.listdir(image_dir)
        match split:
            case "train":
                self.image_names = self.get_case_file_names(filename=train_file)
            case "val":
                self.image_names = self.get_case_file_names(filename=val_file)
            case _:
                raise 'Unrecognized split, use "train" or "val"'

    def get_case_file_names(self, filename: str) -> list:
        df = pd.read_csv(filename, header=None)
        filename_column = df[0]
        filenames = filename_column.to_list()
        return filenames

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        img_name = os.path.join(self.image_dir, image_name + ".jpg")
        mask_name = os.path.join(self.mask_dir, image_name + ".png")

        image = Image.open(img_name)
        mask = Image.open(mask_name)

        if self.image_transforms:
            image = self.image_transforms(image)
        if self.mask_transforms:
            mask = self.mask_transforms(mask)

        return image, mask
