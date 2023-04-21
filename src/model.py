from typing import Any, Callable, List, Optional, Tuple, Union
from functools import partial

import math
import torch
import torch.nn as nn
from torch import Tensor
from src.layers import BasicBlock

from src.transformer import (
    TransformerEncoder,
    TransformerOutput,
    Fp32LayerNorm, 
    init_transformer_weights
)


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


class FLAVATransformerWithoutEmbeddings(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        layernorm: nn.Module,
        hidden_size: int = 768,
        weight_init_fn: Optional[Callable] = None,
        initializer_range: float = 0.02,
        num_cls_token: int = 1,
        **kwargs: Any,
    ):
        super().__init__()
        self.encoder = encoder
        self.layernorm = layernorm
        if num_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, num_cls_token, hidden_size))
        else:
            self.cls_token = None

        if weight_init_fn is None:
            weight_init_fn = partial(
                init_transformer_weights, initializer_range=initializer_range
            )

        self.apply(weight_init_fn)

    def forward(
        self,
        hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> TransformerOutput:
        if hidden_states is None:
            raise ValueError("You have to specify hidden_states")

        if self.cls_token is not None:
            batch_size = hidden_states.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)

        encoder_output = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            return_hidden_states=True,
            return_attn_weights=True,
        )
        sequence_output = encoder_output.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        return TransformerOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )
    

def flava_multimodal_encoder(
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        dropout: float = 0.0,
        intermediate_size: int = 3072,
        intermediate_activation: Callable[..., nn.Module] = nn.GELU,
        layer_norm_eps: float = 1e-12,
        num_cls_token: int = 1,
    ) -> FLAVATransformerWithoutEmbeddings:
        encoder = TransformerEncoder(
            n_layer=num_hidden_layers,
            d_model=hidden_size,
            n_head=num_attention_heads,
            dim_feedforward=intermediate_size,
            activation=intermediate_activation,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            norm_first=True,
        )
        layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)

        return FLAVATransformerWithoutEmbeddings(
            encoder=encoder, 
            layernorm=layernorm, 
            hidden_size=hidden_size,
            num_cls_token=num_cls_token,
        )


class FlavaFusionTransfomer(nn.Module):
    def __init__(self, 
                 # prediction specific parameters
                 out_dim: int = 1,
                 num_classes: int = 1000,
                 # Multimodal encoder specific parameters
                 image_hidden_size: int = 768,
                 text_hidden_size: int = 768,
                 # Multimodal encoder specific parameters
                 multimodal_hidden_size: int = 768,
                 multimodal_num_attention_heads: int = 12,
                 multimodal_num_hidden_layers: int = 6,
                 multimodal_dropout: float = 0.0,
                 multimodal_intermediate_size: int = 3072,
                 multimodal_intermediate_activation: Callable[..., nn.Module] = nn.GELU,
                 multimodal_layer_norm_eps: float = 1e-12,
                **kwargs: Any,):
        
        super().__init__()

        self.mm_encoder = flava_multimodal_encoder(
            hidden_size=multimodal_hidden_size,
            num_attention_heads=multimodal_num_attention_heads,
            num_hidden_layers=multimodal_num_hidden_layers,
            dropout=multimodal_dropout,
            intermediate_size=multimodal_intermediate_size,
            intermediate_activation=multimodal_intermediate_activation,
            layer_norm_eps=multimodal_layer_norm_eps,
            num_cls_token=out_dim
        )

        self.image_to_mm_projection = nn.Linear(image_hidden_size, multimodal_hidden_size)
        self.text_to_mm_projection = nn.Linear(text_hidden_size, multimodal_hidden_size)

        self.output_layers = [nn.Linear(multimodal_hidden_size, num_classes) for i in range(out_dim)]
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):

        image_features = self.image_to_mm_projection(image_features)
        text_features = self.text_to_mm_projection(text_features)
        multimodal_features = torch.cat((image_features, text_features), dim=1)
        multimodal_features = self.mm_encoder(multimodal_features)

        hidden_state = multimodal_features.last_hidden_state
        
        out_list = []
        for i, fc in enumerate(self.output_layers):
            out_list.append(fc(hidden_state[:, i]))
        
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
        
