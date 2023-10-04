import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datamodules.custom_modules import VOC2012SegmentationDataset

with open("/home/haim/code/voc_segmentation/utils/parameters.yaml", "r") as yaml_file:
    parameters = yaml.safe_load(yaml_file)

DATA_DIR = parameters["data_dir"]
image_dir = "/home/haim/hdd/data/voc/VOCdevkit/VOC2012/JPEGImages"
mask_dir = "/home/haim/hdd/data/voc/VOCdevkit/VOC2012/SegmentationObject"
train_file = "/home/haim/hdd/data/voc/VOCdevkit/VOC2012/ImageSets/Layout/train.txt"

train_dataset = VOC2012SegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    split="train",
    transform=None,
    train_file=train_file,
)
print(train_dataset.train_image_filenames)
