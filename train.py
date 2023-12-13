"""
Main script for training a MobileNetV2-based segmentation model on VOC2012.
"""
import yaml
from PIL.Image import NEAREST

from torchvision.transforms import Compose, ToTensor, Resize, Lambda

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from datamodules.datamodule import VOC2012SegmentationDataModule
from models.mobilenet import MobileNetV2Segmentation
from models.unet import PlUNet


with open("parameters.yaml", "r") as file:
    data = yaml.safe_load(file)

IMAGE_DIR = data["image_dir"]
MASK_DIR = data["mask_dir"]
TRAIN_FILE = data["train_file"]
VAL_FILE = data["val_file"]
MAX_EPOCHS = data["max_epochs"]
BATCH_SIZE = data["batch_size"]
NUM_WORKERS = data["num_workers"]
NUM_CLASSES = data["num_classes"]

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
        Lambda(lambda x: x.squeeze(0)),
    ]
)

datamodule = VOC2012SegmentationDataModule(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_transforms=image_transforms,
    mask_transforms=mask_transforms,
    train_file=TRAIN_FILE,
    val_file=VAL_FILE,
)

model = PlUNet(n_channels=3, n_classes=20)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu",
    devices=1,
    fast_dev_run=True,
)


if __name__ == "__main__":
    trainer.fit(model, datamodule)
