"""
Main script for training a MobileNetV2-based segmentation model on VOC2012.
"""
import yaml
import pytorch_lightning as pl
from models import PlUNet
from datamodules import VOC2012SegmentationDataModule
from transforms import image_transforms, mask_transforms

with open("parameters.yaml", "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

IMAGE_DIR = data["image_dir"]
MASK_DIR = data["mask_dir"]
TRAIN_FILE = data["train_file"]
VAL_FILE = data["val_file"]
MAX_EPOCHS = data["max_epochs"]
BATCH_SIZE = data["batch_size"]
NUM_WORKERS = data["num_workers"]
NUM_CLASSES = data["num_classes"]


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
    fast_dev_run=False,
)


if __name__ == "__main__":
    trainer.fit(model, datamodule)
    trainer.save_checkpoint("model.ckpt")
    trainer.test(model, datamodule)
