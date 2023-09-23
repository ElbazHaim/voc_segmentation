# import pytorch_lightning as pl
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
# from torchvision import transforms as T
# from torchvision.transforms import functional as F
# from torchvision.transforms.v2 import PILToTensor
# from torchvision.datasets import VOCSegmentation

# import numpy as np
# import torch

# import numpy as np
# import torch
# from torchvision import transforms as T
# from torchvision.transforms import functional as F


# class VOCSegDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         data_dir,
#         year,
#         batch_size,
#         num_workers,
#         train_split=50000,
#         validation_split=10000,
#     ):
#         super().__init__()
#         self.data_dir = data_dir
#         self.year = year
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.train_split = train_split
#         self.validation_split = validation_split
#         self.setup()

#     def prepare_data(self):
#         VOCSegmentation(
#             root=self.data_dir, year=self.year, image_set="train", download=False
#         ),
#         VOCSegmentation(
#             root=self.data_dir, year=self.year, image_set="val", download=False
#         )
#         VOCSegmentation(
#             root=self.data_dir, year=self.year, image_set="test", download=False
#         )

#     def setup(self, stage=None):
#         self.prepare_data()
#         self.train_ds = (
#             VOCSegmentation(
#                 root=self.data_dir,
#                 year=self.year,
#                 image_set="train",
#                 transform=PILToTensor(),
#                 download=False
#             ),
#         )
#         self.val_ds = VOCSegmentation(
#             root=self.data_dir,
#             year=self.year,
#             image_set="val",
#             transform=PILToTensor(),
#             download=False
#         )
#         self.test_ds = VOCSegmentation(
#             root=self.data_dir,
#             year=self.year,
#             image_set="test",
#             transform=PILToTensor(),
#             download=False
#         )

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_ds,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=True,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_ds,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_ds,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#         )

import numpy as np

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

# from torchvision import transforms
# from torchvision.transforms import functional as F
# from torchvision import transforms as T
# from torchvision.transforms import functional as F
from torchvision.transforms import Resize

import segmentation_transforms as transforms

# F.pil
# from torchvision.transforms.v2 import PILToTensor
from torchvision.datasets import VOCSegmentation
import torch


class VOCSegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        year,
        batch_size,
        num_workers,
        train_split=50000,
        validation_split=10000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.year = year
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.validation_split = validation_split
        self.setup()

    def prepare_data(self):
        # Download the VOC Segmentation dataset if needed
        VOCSegmentation(
            root=self.data_dir, year=self.year, image_set="train", download=False
        )
        VOCSegmentation(
            root=self.data_dir, year=self.year, image_set="val", download=False
        )
        VOCSegmentation(
            root=self.data_dir, year=self.year, image_set="test", download=False
        )

    def setup(self, stage=None):
        self.prepare_data()
        self.transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.CenterCrop(255),
                transforms.Normalize(0.5, 0.25)
                # transforms.RandomResize(min_size=200) #
                # Replace 'height' and 'width' with your desired image dimensions
            ]
        )  # Transformation to convert PIL images to tensors

    def train_dataloader(self):
        train_ds = VOCSegmentation(
            root=self.data_dir,
            year=self.year,
            image_set="train",
            transforms=self.transform,
            download=False,  # Set this to False since we've already downloaded the data
        )
        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        val_ds = VOCSegmentation(
            root=self.data_dir,
            year=self.year,
            image_set="val",
            transform=self.transform,
            download=False,  # Set this to False since we've already downloaded the data
        )
        return DataLoader(
            val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        test_ds = VOCSegmentation(
            root=self.data_dir,
            year=self.year,
            image_set="test",
            transform=self.transform,
            download=False,  # Set this to False since we've already downloaded the data
        )
        return DataLoader(
            test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
