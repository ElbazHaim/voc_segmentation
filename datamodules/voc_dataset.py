import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
# from .custom_transforms import ToTensor, FixedResize
import torchvision
from torchvision.transforms.functional import to_tensor, resize
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import numpy as np


class VOC_Dataset(Dataset):
    def __init__(self, root_dir=None, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split

        self.transform = transform
        self.image_list = []
        self.label_list = []

        # Define the class mapping for VOC
        self.class_map = {
            "background": 0,
            "aeroplane": 1,
            "bicycle": 2,
            "bird": 3,
            "boat": 4,
            "bottle": 5,
            "bus": 6,
            "car": 7,
            "cat": 8,
            "chair": 9,
            "cow": 10,
            "diningtable": 11,
            "dog": 12,
            "horse": 13,
            "motorbike": 14,
            "person": 15,
            "pottedplant": 16,
            "sheep": 17,
            "sofa": 18,
            "train": 19,
            "tvmonitor": 20,
        }

        if self.split == "train":
            train_filenames_txt_path = os.path.join(
                root_dir, "ImageSets", "Segmentation", "train.txt"
            )
            with open(train_filenames_txt_path, "r") as file:
                train_image_names = set([line.strip() for line in file.readlines()])

            image_dir = os.path.join(root_dir, "JPEGImages")

            train_image_filenames = [
                image_filename
                for image_filename in os.listdir(image_dir)
                if image_filename.rstrip(".jpg") in train_image_names
            ]
            self.image_list = [
                os.path.join(image_dir, img) for img in train_image_filenames
            ]

            label_dir = os.path.join(root_dir, "SegmentationObject")
            label_filenames = [
                label_filename
                for label_filename in os.listdir(label_dir)
                if label_filename.rstrip(".png") in train_image_names
            ]
            self.label_list = [
                os.path.join(label_dir, label) for label in label_filenames
            ]

        elif self.split == "test":
            image_dir = os.path.join(root_dir, "JPEGImages")
            self.image_list = [
                os.path.join(image_dir, img) for img in os.listdir(image_dir)
            ]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])

        if self.split == "train":
            label = torchvision.io.read_image(
                self.label_list[idx]
            )  # Image.open(self.label_list[idx])
            label = self.encode_segmap(label)

        if self.transform:
            image = self.transform(image)

            if self.split == "train":
                label = self.transform(label)

        if self.split == "train":
            return image, label
        else:
            return image

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    # def encode_segmap(self, label):
    #     # Convert VOC labels to class indices
    #     label = label.convert("RGB")
    #     label_data = torch.zeros(label.size[1], label.size[0])
    #     for k, v in self.class_map.items():
    #         mask = ((label == torch.tensor(list(map(int, k))))).all(dim=2)
    #         label_data[mask] = v
    #     return label_data.long()
    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        # mask.convert("RGB")
        # mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask


class VOCSegDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, year, batch_size, num_workers):
        super().__init__()
        self.root_dir = data_dir
        self.data_dir = data_dir + "/VOCdevkit/VOC2012"
        self.year = year
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.setup()

    def prepare_data(self):
        # Download the VOC Segmentation dataset if needed
        VOCSegmentation(
            root=self.root_dir, year=self.year, image_set="train", download=False
        )

    #     VOCSegmentation(
    #         root=self.data_dir, year=self.year, image_set="val", download=False
    #     )
    #     # VOCSegmentation(
    #     #     root=self.data_dir, year=self.year, image_set="test", download=False
    #     # )

    def setup(self, stage=None):
        self.prepare_data()
        self.transform = transforms.Compose(
            [
                # torchvision.transforms.functional. # ToTensor(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(255),
                # FixedResize(255),
                # transforms.PILToTensor(),
                torchvision.transforms.CenterCrop(255),
                # transforms.Normalize(0.5, 0.25)
                # transforms.RandomResize(min_size=200) #
                # Replace 'height' and 'width' with your desired image dimensions
            ]
        )  # Transformation to convert PIL images to tensors

    def train_dataloader(self):
        train_ds = VOC_Dataset(
            root_dir=self.data_dir, split="train", transform=self.transform
        )
        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        val_ds = VOC_Dataset(
            root_dir=self.data_dir, split="val", transform=self.transform
        )
        return DataLoader(
            val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    VOCSegDataModule(
        data_dir="/home/haim/hdd/data/voc/2012",
        year="2012",
        batch_size=32,
        num_workers=3,
    )
