"""
Main script for training a MobileNetV2-based segmentation model on VOC2012.
"""
from PIL.Image import NEAREST

from torchvision.transforms import Compose, ToTensor, Resize, Lambda

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from datamodules.datamodule import VOC2012SegmentationDataModule
from models.mobilenet import MobileNetV2Segmentation

IMAGE_DIR = "/home/haim/hdd/data/voc/VOCdevkit/VOC2012/JPEGImages"
MASK_DIR = "/home/haim/hdd/data/voc/VOCdevkit/VOC2012/SegmentationObject"
TRAIN_FILE = (
    "/home/haim/hdd/data/voc/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
)
VAL_FILE = "/home/haim/hdd/data/voc/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
MAX_EPOCHS = 2
BATCH_SIZE = 4
NUM_WORKERS = 2

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

model = MobileNetV2Segmentation(num_classes=20)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu",
    devices=1,
    fast_dev_run=True,
)


if __name__ == "__main__":
    trainer.fit(model, datamodule)
