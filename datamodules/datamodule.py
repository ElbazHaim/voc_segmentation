"""
Data module for VOC2012SegmentationDataset.
"""
import pytorch_lightning as pl
from PIL.Image import NEAREST
from torch import Generator
from torch.utils.data import DataLoader, random_split
from .dataset import VOC2012SegmentationDataset


class VOC2012SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        num_workers,
        image_dir,
        mask_dir,
        image_transforms,
        mask_transforms,
        train_file=None,
        val_file=None,
        trainval_file=None,
        random_seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.train_file = train_file
        self.val_file = val_file
        self.trainval_file = trainval_file
        self.random_seed = random_seed
        self.prepare_datasets()

    def prepare_datasets(self):
        self.setup()

    def setup(self, stage=None):
        dataset = VOC2012SegmentationDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            split="trainval",
            image_transforms=self.image_transforms,
            mask_transforms=self.mask_transforms,
            trainval_file=self.trainval_file,
        )
        train_dataset, val_dataset, test_dataset = random_split(
            dataset=dataset,
            lengths=[0.6, 0.2, 0.2],
            generator=Generator().manual_seed(self.random_seed),
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
