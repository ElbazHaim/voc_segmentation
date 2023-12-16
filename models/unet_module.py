"""
Pytorch lightning module for U-Net.
"""
import torch
import pytorch_lightning as pl
import torchmetrics
from .unet_torch import UNet


class PlUNet(pl.LightningModule):
    def __init__(
        self, n_channels, n_classes, learning_rate=1e-3, bilinear=False
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.bilinear = bilinear
        self.unet = UNet(n_channels, n_classes, bilinear)

    def forward(self, x):
        return self.unet(x)

    def common_step(self, batch, batch_idx):
        inputs, masks = batch
        outputs = self(inputs)
        masks = masks.type(torch.long)
        loss = torch.nn.functional.cross_entropy(outputs, masks)
        accuracy = torchmetrics.functional.accuracy(
            outputs, masks, task="multiclass", num_classes=20
        )
        jaccard = torchmetrics.functional.jaccard_index(
            outputs, masks, task="multiclass", num_classes=20
        )

        return loss, accuracy, jaccard

    def training_step(self, batch, batch_idx):
        loss, accuracy, jaccard = self.common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": accuracy,
                "train_jaccard": jaccard,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, jaccard = self.common_step(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": accuracy,
                "val_jaccard": jaccard,
            }
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, jaccard = self.common_step(batch, batch_idx)
        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": accuracy,
                "test_jaccard": jaccard,
            }
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
