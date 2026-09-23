import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DAG(nn.Module):
    """Dual-attention gate from Chen et al., equations 15–17."""

    def __init__(self, channels):
        super().__init__()

        self.branch_average = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.branch_maximum = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.output = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        f1 = torch.sigmoid(self.branch_average(x))
        f2 = torch.sigmoid(self.branch_maximum(x))

        g1 = F.adaptive_avg_pool2d(f1, 1)
        g2 = F.adaptive_max_pool2d(f2, 1)

        return self.output(x * g1 + x * g2)


class MFEF(nn.Module):
    """Multi-scale feature enhancement and fusion, equations 18–22."""

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)

        self.conv3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.conv5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
        )

        self.gap_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

        self.compress = nn.Conv2d(
            channels * 4,
            channels,
            kernel_size=1,
        )

        self.dilated = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
        )

    def forward(self, x):
        size = x.shape[-2:]

        global_feature = self.gap_projection(
            F.adaptive_avg_pool2d(x, 1)
        )

        global_feature = F.interpolate(
            global_feature,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        multi_scale = torch.cat(
            [
                self.conv1(x),
                self.conv3(x),
                self.conv5(x),
                global_feature,
            ],
            dim=1,
        )

        m1 = F.relu(self.compress(multi_scale), inplace=True)
        w1 = torch.softmax(m1, dim=1)

        m2 = F.relu(self.dilated(x), inplace=True)
        w2 = torch.softmax(m2, dim=1)

        weights = 0.5 * w1 + 0.5 * w2
        return x * weights


class AMFUConfig(PretrainedConfig):
    model_type = "amfu-net"

    def __init__(
        self,
        base_channels=32,
        image_size=256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.base_channels = base_channels
        self.image_size = image_size


class AMFUNet(PreTrainedModel):
    config_class = AMFUConfig

    def __init__(self, config):
        super().__init__(config)

        b = config.base_channels

        self.enc1 = ConvBNReLU(1, b)
        self.enc2 = ConvBNReLU(b, b * 2)
        self.enc3 = ConvBNReLU(b * 2, b * 4)
        self.enc4 = ConvBNReLU(b * 4, b * 8)

        self.bottleneck = ConvBNReLU(b * 8, b * 16)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.mfef = MFEF(b * 16)

        self.dag1 = DAG(b)
        self.dag2 = DAG(b * 2)
        self.dag3 = DAG(b * 4)
        self.dag4 = DAG(b * 8)

        self.up4 = nn.ConvTranspose2d(
            b * 16,
            b * 8,
            kernel_size=2,
            stride=2,
        )
        self.dec4 = ConvBNReLU(b * 16, b * 8)

        self.up3 = nn.ConvTranspose2d(
            b * 8,
            b * 4,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = ConvBNReLU(b * 8, b * 4)

        self.up2 = nn.ConvTranspose2d(
            b * 4,
            b * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = ConvBNReLU(b * 4, b * 2)

        self.up1 = nn.ConvTranspose2d(
            b * 2,
            b,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = ConvBNReLU(b * 2, b)

        self.head = nn.Conv2d(
            b,
            1,
            kernel_size=1,
        )

        self.post_init()

    def forward(self, pixel_values, labels=None):
        s1 = self.enc1(pixel_values)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        x = self.mfef(
            self.bottleneck(self.pool(s4))
        )

        x = self.up4(x)
        x = self.dec4(
            torch.cat([x, self.dag4(s4)], dim=1)
        )

        x = self.up3(x)
        x = self.dec3(
            torch.cat([x, self.dag3(s3)], dim=1)
        )

        x = self.up2(x)
        x = self.dec2(
            torch.cat([x, self.dag2(s2)], dim=1)
        )

        x = self.up1(x)
        x = self.dec1(
            torch.cat([x, self.dag1(s1)], dim=1)
        )

        logits = self.head(x)

        loss = None

        if labels is not None:
            probabilities = torch.sigmoid(logits)
            dimensions = (1, 2, 3)

            intersection = (
                probabilities * labels
            ).sum(dimensions)

            dice = (
                2 * intersection + 1.0
            ) / (
                probabilities.sum(dimensions)
                + labels.sum(dimensions)
                + 1.0
            )

            loss = (
                F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                )
                + (1 - dice.mean())
            )

        return {
            "loss": loss,
            "logits": logits,
        }