import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class VOC2012SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        split="train",
        transform=None,
        train_file="",
        val_file="",
    ):
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(image_dir)
        self.mask_dir = os.path.join(mask_dir)
        self.image_files = os.listdir(image_dir)
        match split:
            case "train":
                self.image_names = self.get_case_file_names(
                    filename=train_file)
            case "val":
                self.get_case_file_names(filename=val_file)
            case _:
                raise "No split case chosen"

    def get_case_file_names(self, filename: str) -> list:
        df = pd.read_csv(filename, delimiter=' ', header=None)
        filename_column = df[0]
        filenames = filename_column.to_list()
        return filenames

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        img_name = os.path.join(self.image_dir, image_name + ".jpg")
        mask_name = os.path.join(self.mask_dir, image_name + ".png")

        image = Image.open(img_name)
        mask = Image.open(mask_name)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
