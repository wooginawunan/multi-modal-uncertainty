import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class OutputLayer(nn.Module):
    def __init__(self, in_features, output_shape, fullconv=False, activation="log_softmax"):
        super(OutputLayer, self).__init__()
        if not isinstance(output_shape, (list, tuple)):
            output_shape = [output_shape]
        self.output_shape = output_shape
        self.flattened_output_shape = int(np.prod(output_shape))
        
        self.activation = activation
        self.fullconv = fullconv
        if self.fullconv:
            self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=self.flattened_output_shape, kernel_size=1)
        else:
            self.fc_layer = nn.Linear(in_features, self.flattened_output_shape)

    def forward(self, x, use_activation=True):
        if self.fullconv:
            h = self.conv1(x)
        else:
            h = self.fc_layer(x)
        if len(self.output_shape) > 1:
            h = h.view(h.shape[0], *self.output_shape)
        if use_activation:
            if self.activation == "log_softmax":
                h = F.log_softmax(h, dim=-1)
            else:
                raise KeyError(self.activation)
        return h

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckV2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.conv3(out)

        out += residual

        return out


    """Add gaussian noise across all 4 views"""

    def __init__(self, gaussian_noise_std):
        super(TwoViewsGaussianNoise, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std

    def forward(self, x):
        if not self.gaussian_noise_std or not self.training:
            return x

        return {
            "cc": self._add_gaussian_noise(x["cc"]),
            "mlo": self._add_gaussian_noise(x["mlo"])
        }

    def _add_gaussian_noise(self, single_view):
        return single_view + single_view.new(single_view.shape).normal_(std=self.gaussian_noise_std)