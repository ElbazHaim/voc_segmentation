import torch
import torch.nn as nn
import pytorch_lightning as pl


class DepthwiseConvolution(nn.Module):
    def __init__(self, size, stride=(1, 1)):
        super().__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(size, size, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            nn.BatchNorm2d(size),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_image):
        x = self.depth_conv(input_image)
        return x


class PointwiseConvolution(nn.Module):
    def __init__(self, in_size, out_size, stride=(1, 1)):
        super().__init__()
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=(3, 3), stride=stride, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_image):
        x = self.point_conv(input_image)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_size, out_size, expansion, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, expansion * in_size, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(expansion * in_size),
            nn.ReLU(inplace=True),
        )
        self.depthwise = DepthwiseConvolution(
            expansion * in_size, stride=(stride, stride)
        )
        self.pointwise = PointwiseConvolution(expansion * in_size, out_size)

        self.expansion = expansion
        self.stride = stride

    def forward(self, input_image):
        if self.expansion == 1:
            x = self.depthwise(input_image)
            x = self.pointwise(x)
        else:
            x = self.conv1(input_image)
            x = self.depthwise(x)
            x = self.pointwise(x)

        if self.stride == 1:
            x = input_image + x

        return x


class MobileNetV2Segmentation(pl.LightningModule):
    def __init__(self, num_classes):
        super(MobileNetV2Segmentation, self).__init__()

    def forward(self, x):
        features = self.mobilenet(x)
        segmentation = self.segmentation_head(features)
        return segmentation

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
