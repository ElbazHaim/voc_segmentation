import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from .model_utils.parts import DoubleConv, Down, Up

# from voc_dataset import VOC_Dataset

# semantic_dataset = VOC_Dataset


class UNet(nn.Module):
    def __init__(self, num_classes=19, bilinear=False):
        super().__init__()
        self.bilinear = bilinear
        self.num_classes = num_classes
        self.layer1 = DoubleConv(3, 64)
        self.layer2 = Down(64, 128)
        self.layer3 = Down(128, 256)
        self.layer4 = Down(256, 512)
        self.layer5 = Down(512, 1024)

        self.layer6 = Up(1024, 512, bilinear=self.bilinear)
        self.layer7 = Up(512, 256, bilinear=self.bilinear)
        self.layer8 = Up(256, 128, bilinear=self.bilinear)
        self.layer9 = Up(128, 64, bilinear=self.bilinear)

        self.layer10 = nn.Conv2d(64, self.num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        x6 = self.layer6(x5, x4)
        x6 = self.layer7(x6, x3)
        x6 = self.layer8(x6, x2)
        x6 = self.layer9(x6, x1)

        return self.layer10(x6)


class SegModel(pl.LightningModule):
    def __init__(self):
        super(SegModel, self).__init__()
        self.example_input_array = torch.rand([6, 3, 255, 255])
        self.batch_size = 4
        self.learning_rate = 1e-3
        #         self.net = torchvision.models.segmentation.fcn_resnet50(pretrained = False, progress = True, num_classes = 19)
        self.net = UNet(num_classes=19, bilinear=False)
        #         self.net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = False, progress = True, num_classes = 19)
        # self.net = ENet(num_classes = 19)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.35675976, 0.37380189, 0.3764753],
                    std=[0.32064945, 0.32098866, 0.32325324],
                ),
            ]
        )
        # self.trainset = semantic_dataset(split="train", transform=self.transform)
        # self.testset = semantic_dataset(split="test", transform=self.transform)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        # print(mask[:, 0, :, :].shape)
        mask = mask[:, 0, :, :]
        # import sys
        # sys.exit()
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        self.log_dict({"train_loss": loss_val}, on_step=True)
        return {"loss": loss_val}

    def validation_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        # print(mask[:, 0, :, :].shape)
        mask = mask[:, 0, :, :]
        # import sys
        # sys.exit()
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        self.log_dict(
            {"val_loss": loss_val}, on_step=False, prog_bar=False, on_epoch=True
        )
        return {"loss": loss_val}

    # def test_step(self, batch, batch_nb):
    #     img, mask = batch
    #     img = img.float()
    #     mask = mask.long()
    #     out = self.forward(img)
    #     # print(mask[:, 0, :, :].shape)
    #     mask = mask[:, 0, :, :]
    #     # import sys
    #     # sys.exit()
    #     loss_val = F.cross_entropy(out, mask, ignore_index=250)
    #     self.log_dict({"test_loss": loss_val})
    #     return {"loss": loss_val}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=1, shuffle=True)


# model = SegModel()
# trainer = pl.Trainer(accelerator="gpu", max_epochs=2, fast_dev_run=True)
# trainer.fit(model, datamodule)
