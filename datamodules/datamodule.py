import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MnistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
        train_split=50000,
        validation_split=10000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.validation_split = validation_split

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = MNIST(
            root=self.data_dir,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
            download=False,
        )
        self.train_ds, self.val_ds = random_split(
            entire_dataset, [self.train_split, self.validation_split]
        )
        self.test_ds = MNIST(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
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
