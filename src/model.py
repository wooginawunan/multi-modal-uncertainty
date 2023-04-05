import math
import torch
import torch.nn as nn

from .layers import BasicBlock

class ResNet(nn.Module):
    def __init__(self, num_channels, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        #self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
   
class MultiHeadFC(nn.Module):
    def __init__(self, input_dim, num_classes, out_dim):
        super(MultiHeadFC, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dim, num_classes*out_dim)

    def forward(self, x):
        out = self.fc(x)
        # The shape of `outputs` is (batch_size, num_classes * ensemble_size).
        out_list = torch.split(out, self.num_classes, dim=-1) 
        out = torch.stack(out_list, dim=1) # (batch_size, ensemble_size, num_classes)
        
        return out
    
class MIMOResNet(ResNet):
    def __init__(self, num_channels, emb_dim, out_dim, num_classes):
        
        input_dim = num_channels * emb_dim
        super(MIMOResNet, self).__init__(input_dim, BasicBlock, [2, 2, 2])
        self.output_layer = MultiHeadFC(128 * BasicBlock.expansion, num_classes, out_dim)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # x: B, E, C, H, W
        x = x.view(x.size(0), -1,  x.size(3), x.size(4)) # x: B, E*C, H, W
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 
        out = self.output_layer(x) # x: B, E, C

        return out
    
    def compute_loss(self, y_hat, y):

        assert y.shape[0] == y_hat.shape[0]
        
        y = y.unsqueeze(1).repeat(1, y_hat.shape[1]).view(-1)
        y_hat = y_hat.view(-1, y_hat.shape[2])
        
        return self.loss(y_hat, y)
    
class FusionTransfomer(nn.Module):
    def __init__(self, num_input_channel):
        pass
    def forward(self, x_, target_quarter = None ):
        pass
          