import yaml

from PIL.Image import NEAREST

from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from datamodules.dataset import VOC2012SegmentationDataset

IMAGE_DIR = "/home/haim/hdd/data/voc/VOCdevkit/VOC2012/JPEGImages"
MASK_DIR = "/home/haim/hdd/data/voc/VOCdevkit/VOC2012/SegmentationObject"
TRAIN_FILE = (
    "/home/haim/hdd/data/voc/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
)


if __name__ == "__main__":
    print("hi")
