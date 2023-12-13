"""

"""
from pathlib import Path
import yaml
import optuna
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from models.unet import PlUNet
from transforms import image_transforms, mask_transforms
from datamodules.datamodule import VOC2012SegmentationDataModule


with open("parameters.yaml", "r") as file:
    params = yaml.safe_load(file)

IMAGE_DIR = params["image_dir"]
MASK_DIR = params["mask_dir"]
TRAIN_FILE = params["train_file"]
VAL_FILE = params["val_file"]
MAX_EPOCHS = params["max_epochs"]
BATCH_SIZE = params["batch_size"]
NUM_WORKERS = params["num_workers"]
NUM_CLASSES = params["num_classes"]
N_TRIALS = params["n_trials"]
TIMEOUT = params["timeout"]
INPUT_SHAPE = tuple(params["input_shape"])


def objective(trial: optuna.trial.Trial) -> float:
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-9, 1e-3)

    model = PlUNet(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        learning_rate=learning_rate,
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

    trainer = pl.Trainer(
        fast_dev_run=False,
        logger=True,
        enable_checkpointing=True,
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=3,
                verbose=False,
                mode="min",
            )
        ],
    )
    hyperparameters = dict(learning_rate=learning_rate)
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    prune = False
    pruner = optuna.pruners.MedianPruner() if prune else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)
