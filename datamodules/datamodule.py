import pytorch_lightning as pl
from PIL.Image import NEAREST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from dataset import VOC2012SegmentationDataset


class VOC2012SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        num_workers,
        image_dir,
        mask_dir,
        image_transforms,
        mask_transforms,
        train_file,
        val_file,
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

    def prepare_data(self):
        pass

    def setup(self, stage=None):
            self.train_dataset = VOC2012SegmentationDataset(
                image_dir=self.image_dir,
                mask_dir=self.mask_dir,
                split="train",
                image_transforms=self.image_transforms,
                mask_transforms=self.mask_transforms,
                train_file=self.train_file,
            )
            self.val_dataset = VOC2012SegmentationDataset(
                image_dir=self.image_dir,
                mask_dir=self.mask_dir,
                split="val",
                image_transforms=self.image_transforms,
                mask_transforms=self.mask_transforms,
                val_file=self.val_file,
            )
            
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    image_transforms = Compose(
        [
            ToTensor(),
            Resize((374, 500), antialias=True),
        ]
    )

    mask_transforms = Compose(
        [
            ToTensor(),
            Resize((374, 500), interpolation=NEAREST, antialias=True),
        ]
    )
