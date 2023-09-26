import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from datamodules.voc_dataset import VOCSegDataModule
from utils._utils import parameters  # , _get_dev_run

from models.segmodel import SegModel


if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", "mnist_model_v0")
    # fast_dev_run = _get_dev_run()
    fast_dev_run = True
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

    # model = MobileNetV2Segmentation(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)

    model = SegModel()
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        max_epochs=NUM_EPOCHS,
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(model, datamodule)
    # trainer.validate(model, datamodule)
    # trainer.test(model, datamodule)

    model.to_onnx("segmentation_model.onnx", export_params=True)
