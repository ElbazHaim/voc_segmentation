import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, batch_size, input_size, num_classes):
        super().__init__()
        self.example_input_array = torch.randn([batch_size, input_size])
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.decoder.forward(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss, x_hat, z, x, y

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
