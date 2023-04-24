from typing import Any, Callable, List, Optional, Tuple, Union
from functools import partial

import math
import torch
import torch.nn as nn
from src.layers import BasicBlock

model_configure ={
    "Vanilla": (4, 1),
    "MIMO-shuffle-instance": (4, 4),
    "MIMO-shuffle-view": (4, 4),
    "MultiHead": (4, 4),
    "MIMO-shuffle-all": (4, 4),
    "single-model-weight-sharing": (1, 1)
}

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
        if len(x.shape) == 5:
            # concatenate the view/ensemble dimension as the channel dimension
            x = x.view(x.size(0), -1,  x.size(3), x.size(4)) # x: B, E*C, H, W
        else:
            # sharinng weight model
            # B*E, C, H, W
            pass 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 
        out = self.output_layer(x) # x: B, E, C

        return out
    
    def compute_loss(self, y_hat, y, eval=False):

        assert y.shape[0] == y_hat.shape[0]
        
        y = y.view(-1)
        if not eval:
            y_hat = y_hat.view(-1, y_hat.shape[2])
        else:
            y_hat = y_hat.mean(1)

        return self.loss(y_hat, y)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, B, num_head, N, c
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attention:
            return x, attn
        else:
            return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_after=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop = nn.Dropout(drop_after)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=return_attention)
            x = x + self.drop(y)
            x = x + self.drop(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop(self.attn(self.norm1(x)))
            x = x + self.drop(self.mlp(self.norm2(x)))
            return x

class Transformer(nn.Module):
    def __init__(self, width, layers, heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_after=0., ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.blocks = nn.ModuleList([
            Block(
                dim=width, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_after=drop_after, 
                norm_layer=nn.LayerNorm)
            for i in range(layers)])

    def forward(self, x: torch.Tensor):
        for i in range(len((self.blocks))):
            x = self.blocks[i](x)
        return x
    
class FlavaFusionTransfomer(nn.Module):
    def __init__(self, 
                 # prediction specific parameters
                 out_dim: int = 1,
                 num_classes: int = 2,
                 # Multimodal encoder specific parameters
                 image_hidden_size: int = 768,
                 text_hidden_size: int = 768,
                 # Multimodal encoder specific parameters
                 multimodal_hidden_size: int = 768,
                 multimodal_num_attention_heads: int = 6,
                 multimodal_num_hidden_layers: int = 6,
                 multimodal_dropout: float = 0.1,
                **kwargs: Any,):
        
        super().__init__()

        self.mm_encoder = Transformer(width=multimodal_hidden_size, 
                                      layers=multimodal_num_hidden_layers, 
                                      heads=multimodal_num_attention_heads,
                                      drop_rate=multimodal_dropout, 
                                      attn_drop_rate=multimodal_dropout, 
                                      drop_after=multimodal_dropout, )
        
        self.ln_pre = nn.LayerNorm(multimodal_hidden_size)
        self.ln_post = nn.LayerNorm(multimodal_hidden_size)

        self.image_to_mm_projection = nn.Linear(image_hidden_size, multimodal_hidden_size)
        self.text_to_mm_projection = nn.Linear(text_hidden_size, multimodal_hidden_size)

        self.output_layers = nn.ModuleList([nn.Linear(multimodal_hidden_size, num_classes) for i in range(out_dim)])
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        image_features, text_features = x

        image_features = self.image_to_mm_projection(image_features)
        text_features = self.text_to_mm_projection(text_features)
        multimodal_features = torch.cat((image_features, text_features), dim=1)

        multimodal_features = self.ln_pre(multimodal_features)
        out = self.mm_encoder(multimodal_features)
        out = self.ln_post(out)
        
        # hidden_state = multimodal_features.last_hidden_state
        
        out_list = []
        for i, fc in enumerate(self.output_layers):
            out_list.append(fc(out[:, i, :]))
        
        out = torch.stack(out_list, dim=1) # (batch_size, ensemble_size, num_classes)

        return out

    def compute_loss(self, y_hat, y, eval=False):

        assert y.shape[0] == y_hat.shape[0]
        
        y = y.view(-1)
        if not eval:
            y_hat = y_hat.view(-1, y_hat.shape[2])
        else:
            y_hat = y_hat.mean(1)

        return self.loss(y_hat, y)
        
