import torch
import torch.nn as nn
import pytorch_lightning as pl
from icecream import ic


class DepthwiseConvolution(nn.Module):
    def __init__(self, size, stride=(1, 1)):
        super().__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(
                size,
                size,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
            ),
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
            nn.Conv2d(
                in_size,
                out_size,
                kernel_size=(3, 3),
                stride=stride,
                bias=False,
                padding="same",
            ),
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
            nn.Conv2d(
                in_size,
                expansion * in_size,
                kernel_size=(1, 1),
                bias=False,
            ),
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
        ic(input_image.shape)
        if self.expansion == 1:
            x = self.depthwise(input_image)
            x = self.pointwise(x)
        else:
            x = self.conv1(input_image)
            x = self.depthwise(x)
            x = self.pointwise(x)

        # if self.stride == 1:
        #     x = input_image + x

        return x


class MobileNetV2Segmentation(pl.LightningModule):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            Bottleneck(32, 16, 1, 1),
            Bottleneck(16, 24, 6, 2),
            Bottleneck(24, 24, 6, 1),
            Bottleneck(24, 32, 6, 2),
            Bottleneck(32, 32, 6, 1),
            Bottleneck(32, 32, 6, 1),
            Bottleneck(32, 32, 6, 1),
            Bottleneck(32, 64, 6, 2),
            Bottleneck(64, 64, 6, 1),
            Bottleneck(64, 64, 6, 1),
            Bottleneck(64, 64, 6, 1),
            Bottleneck(64, 96, 6, 1),
            Bottleneck(96, 96, 6, 1),
            Bottleneck(96, 96, 6, 1),
            Bottleneck(96, 160, 6, 2),
            Bottleneck(160, 160, 6, 1),
            Bottleneck(160, 160, 6, 1),
            Bottleneck(160, 320, 6, 1),
            nn.Conv2d(320, 1280, kernel_size=(1, 1)),
        )

        self.segmentation_head = nn.Sequential(
            # TODO: Implement a segmentation head. Possibly DeepLab?
            # Need something that will fit in 6GB RAM GPU.
        )

    def forward(self, inputs):
        features = self.feature_extractor(inputs)
        predicted_masks = self.segmentation_head(features)
        return predicted_masks

    def training_step(self, batch, batch_idx):
        inputs, masks = batch
        outputs = self(inputs)
        loss = nn.functional.cross_entropy(outputs, masks)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
