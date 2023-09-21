import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import VOCSegmentation


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

    def prepare_data(self):
        VOCSegmentation(
            root=self.data_dir, year=self.year, image_set="train", download=True
        ),
        VOCSegmentation(
            root=self.data_dir, year=self.year, image_set="val", download=True
        )
        VOCSegmentation(
            root=self.data_dir, year=self.year, image_set="test", download=True
        )

    def setup(self, stage):
        self.train_ds = (
            VOCSegmentation(
                root=self.data_dir, year=self.year, image_set="train", download=True
            ),
        )
        self.val_ds = VOCSegmentation(
            root=self.data_dir, year=self.year, image_set="val", download=True
        )
        self.test_ds = VOCSegmentation(
            root=self.data_dir, year=self.year, image_set="test", download=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
