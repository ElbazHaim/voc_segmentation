from segmentation_data_module import VOCSegDataModule
from _utils import parameters

IN_CHANNELS = parameters["in_channels"]
NUM_CLASSES = parameters["num_classes"]
LEARNING_RATE = parameters["learning_rate"]
BATCH_SIZE = parameters["batch_size"]
NUM_EPOCHS = parameters["num_epochs"]
DATA_DIR = parameters["data_dir"]
YEAR = parameters["year"]

datamodule = VOCSegDataModule(
    data_dir=DATA_DIR, year=YEAR, batch_size=BATCH_SIZE, num_workers=3
)

train_ds = datamodule.train_dataloader()
