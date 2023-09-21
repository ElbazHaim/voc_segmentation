import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MobileNetV2Segmentation(pl.LightningModule):
    def __init__(self, in_channels, num_classes):
        super(MobileNetV2Segmentation, self).__init__()
        self.hparams.learning_rate = 0.001
        self.save_hyperparameters()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        return loss, outputs, inputs

    def training_step(self, batch, batch_idx):
        training_loss, _, _, _, _ = self._step(batch, batch_idx)
        self.log("training_loss", training_loss)
        return training_loss

    def validation_step(self, batch, batch_idx):
        validation_loss, _, _, _, _ = self._step(batch, batch_idx)
        self.log("validation_loss", validation_loss)
        return validation_loss

    def test_step(self, batch, batch_idx):
        test_loss, _, _, _, _ = self._step(batch, batch_idx)
        self.log("test_loss", test_loss)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
