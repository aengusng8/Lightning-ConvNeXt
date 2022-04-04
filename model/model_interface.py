# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import pytorch_lightning as pl

from torch import optim as optim
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

from sklearn.metrics import accuracy_score

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False

from model.common import Block, LayerNorm
# from model.optim_factory import LayerDecayValueAssigner, get_parameter_groups


class MInterface(pl.LightningModule):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self,
                 args,
                 in_chans=3,
                 num_classes=1000,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.,
                 ):
        super().__init__()
        self.args = args
        self.configure_loss()
        # stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item()
                    for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # global average pooling, (N, C, H, W) -> (N, C)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y_true = batch

        y_pred = self(x)
        loss = self.loss_function(y_pred, y_true)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = self.loss_function(y_pred, y_true)

        threshold = torch.tensor([0.5]).to(self.device)

        y_pred = (y_pred > threshold).float() * 1
        self.log_dict({'val_loss': loss, 'val_acc': self._multilabel_accuracy(y_true, y_pred)})
        

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = self.loss_function(y_pred, y_true)

        threshold = torch.tensor([0.5]).to(self.device)

        y_pred = (y_pred > threshold).float() * 1
        self.log_dict({'test_loss': loss, 'test_acc': self._multilabel_accuracy(y_true, y_pred)})

    def configure_loss(self):
        self.loss_function = nn.BCEWithLogitsLoss()
        # self.loss_function = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        args = self.args
        opt_lower = args.opt.lower()
        weight_decay = args.weight_decay
        parameters = self.parameters()

        if 'fused' in opt_lower:
            assert has_apex and torch.cuda.is_available(
            ), 'APEX and CUDA required for fused optimizers'

        opt_args = dict(lr=args.lr, weight_decay=weight_decay)
        if hasattr(args, 'opt_eps') and args.opt_eps is not None:
            opt_args['eps'] = args.opt_eps
        if hasattr(args, 'opt_betas') and args.opt_betas is not None:
            opt_args['betas'] = args.opt_betas

        opt_split = opt_lower.split('_')
        opt_lower = opt_split[-1]
        if opt_lower == 'sgd' or opt_lower == 'nesterov':
            opt_args.pop('eps', None)
            optimizer = optim.SGD(
                parameters, momentum=args.momentum, nesterov=True, **opt_args)
        elif opt_lower == 'momentum':
            opt_args.pop('eps', None)
            optimizer = optim.SGD(
                parameters, momentum=args.momentum, nesterov=False, **opt_args)
        elif opt_lower == 'adam':
            optimizer = optim.Adam(parameters, **opt_args)
        elif opt_lower == 'adamw':
            optimizer = optim.AdamW(parameters, **opt_args)
        elif opt_lower == 'nadam':
            optimizer = Nadam(parameters, **opt_args)
        elif opt_lower == 'radam':
            optimizer = RAdam(parameters, **opt_args)
        elif opt_lower == 'adamp':
            optimizer = AdamP(parameters, wd_ratio=0.01,
                              nesterov=True, **opt_args)
        elif opt_lower == 'sgdp':
            optimizer = SGDP(parameters, momentum=args.momentum,
                             nesterov=True, **opt_args)
        elif opt_lower == 'adadelta':
            optimizer = optim.Adadelta(parameters, **opt_args)
        elif opt_lower == 'adafactor':
            if not args.lr:
                opt_args['lr'] = None
            optimizer = Adafactor(parameters, **opt_args)
        elif opt_lower == 'adahessian':
            optimizer = Adahessian(parameters, **opt_args)
        elif opt_lower == 'rmsprop':
            optimizer = optim.RMSprop(
                parameters, alpha=0.9, momentum=args.momentum, **opt_args)
        elif opt_lower == 'rmsproptf':
            optimizer = RMSpropTF(parameters, alpha=0.9,
                                  momentum=args.momentum, **opt_args)
        elif opt_lower == 'novograd':
            optimizer = NovoGrad(parameters, **opt_args)

        return optimizer
    def _multilabel_accuracy(self, y_true, y_pred):
        diff = y_true - y_pred
        diff = torch.sum(torch.abs(diff))
        return 1 - (diff / torch.numel(y_true))


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


def convnext_tiny(args, num_classes, pretrained=True, in_22k=False):
    model = MInterface(args=args, depths=[3, 3, 9, 3], dims=[
                       96, 192, 384, 768], num_classes=1000)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_small(args, num_classes, pretrained=True, in_22k=False, **kwargs):
    model = MInterface(args=args, depths=[3, 3, 27, 3], dims=[
                       96, 192, 384, 768], num_classes=1000)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_base(args, num_classes, pretrained=True, in_22k=False):
    model = MInterface(args=args, depths=[3, 3, 27, 3], dims=[
                       128, 256, 512, 1024], num_classes=1000)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


model_dict = {
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
    "convnext_base": convnext_base,
}


def get_model(args, num_classes, model="convnext_base"):
    model = model_dict[f"{model}"](args=args, num_classes=num_classes)
    model.head = nn.Linear(in_features=model.head.in_features, out_features=num_classes, bias=True)
    return model


# if __name__ == '__main__':
#     model = get_model(num_classes=20)
#     print(model.head)
