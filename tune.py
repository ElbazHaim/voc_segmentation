"""

"""
import yaml
import optuna
import pytorch_lightning as pl
from models import PlUNet
from transforms import image_transforms, mask_transforms
from datamodules import VOC2012SegmentationDataModule


with open("parameters.yaml", "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

IMAGE_DIR = params["image_dir"]
MASK_DIR = params["mask_dir"]
TRAIN_FILE = params["train_file"]
VAL_FILE = params["val_file"]
TRAINVAL_FILE = params["trainval_file"]
MAX_EPOCHS = params["max_epochs"]
BATCH_SIZE = params["batch_size"]
NUM_WORKERS = params["num_workers"]
NUM_CLASSES = params["num_classes"]
N_TRIALS = params["n_trials"]
TIMEOUT = params["timeout"]
INPUT_SHAPE = tuple(params["input_shape"])


def objective(trial: optuna.trial.Trial) -> float:
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-9, 1e-3)
    bilinear = trial.suggest_categorical("bilinear", [True, False])

    model = PlUNet(
        n_channels=3,
        n_classes=NUM_CLASSES,
        learning_rate=learning_rate,
        bilinear=bilinear,
    )

    datamodule = VOC2012SegmentationDataModule(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_transforms=image_transforms,
        mask_transforms=mask_transforms,
        trainval_file=TRAINVAL_FILE,
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
    hyperparameters = dict(learning_rate=learning_rate, bilinear=bilinear)
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    prune = False
    pruner = (
        optuna.pruners.MedianPruner() if prune else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

    best_trial_value = study.best_trial.value
    best_trial_params = study.best_trial.params

    with open("best_params.yaml", "w", encoding="utf-8") as file:
        yaml.dump(best_trial_params, file)
        print("Best params saved to best_params.yaml")

    print("Best trial:")
    print(f"Value: {best_trial_value}")

    print("Params: ")
    for key, value in best_trial_params.items():
        print("{key}: {value}")
