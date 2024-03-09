# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        #print(out.shape, out.shape[1] * out.shape[2]*out.shape[3] / 49)
        return out


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def unfold_layers(layers, x, vs, downsample, bn1ds):
    B = x.size(0)
    features = 0
    for i, layer in enumerate(layers):
        with torch.no_grad():

            x = layer(x)
            
            #x_pool = F.adaptive_avg_pool2d(x, output_size=7)
        # if downsample is not None:
        #     features  = features + bns[i](F.relu(downsample(F.relu(x))).view(B, -1))
        # else:
        #     features  = features + bns[i](F.relu(x).view(B, -1))

        if downsample is not None:
            #features  = features + bns[i](F.relu(downsample(F.relu(x))).view(B, -1))
            features  = features + vs[i] *  (F.adaptive_avg_pool2d(downsample(x), output_size=1).view(B,-1))
            
            assert NotImplementedError

        else:
            #features = features + bns[i](F.adaptive_avg_pool2d(F.relu(x), output_size=1).view(B,-1) )
            #features = features + vs[i] * bn1ds[i](F.adaptive_avg_pool2d(x, output_size=1).view(B,-1))
            features = features + vs[i] * (F.adaptive_avg_pool2d(x, output_size=1).view(B,-1))
            
            #features  = features + bns[i](F.relu(x).view(B, -1))

    return x, features 

class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            zero_init_residual=False,
            groups=1,
            widen=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            normalize=False,
            output_dim=0,
            hidden_mlp=0,
            nmb_prototypes=0,
            eval_mode=False,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(
            3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False
        )
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        # assume input dim is 224
        self.layer1_out_shape = [num_out_filters, 56, 56]
        
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer2_out_shape = [num_out_filters, 28, 28]
        
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer2_out_shape = [num_out_filters, 14, 14]
        
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.layer2_out_shape = [num_out_filters, 7, 7]
        

        #self.layer1_v = nn.Parameter(torch.zeros(1, 2048, 7, 7))
        #self.layer2_v = nn.Parameter(torch.zeros(1, 2048, 7, 7))
        #self.layer3_v = nn.Parameter(torch.zeros(1, 2048, 7, 7))
        #self.layer4_v = nn.Parameter(torch.zeros(1, 2048, 7, 7))
        

        # self.layer1_v = nn.BatchNorm2d()
        # self.layer2_v = nn.Parameter(torch.zeros(1, 2048, 7, 7))
        # self.layer3_v = nn.Parameter(torch.zeros(1, 2048, 7, 7))
        # self.layer1_v = nn.BatchNorm1d(2048 * 7 * 7)
        # self.layer2_v = nn.BatchNorm1d(2048 * 7 * 7)
        # self.layer3_v = nn.BatchNorm1d(2048 * 7 * 7)
        
        # self.layer3_v_0 = nn.BatchNorm1d(2048 * 7 * 7)
        # self.layer3_v_1 = nn.BatchNorm1d(2048 * 7 * 7)
        # self.layer3_v_2 = nn.BatchNorm1d(2048 * 7 * 7)
        # self.layer3_v_3 = nn.BatchNorm1d(2048 * 7 * 7)
        # self.layer3_v_4 = nn.BatchNorm1d(2048 * 7 * 7)
        # self.layer3_v_5 = nn.BatchNorm1d(2048 * 7 * 7)
        
        # self.layer4_v_0 = nn.BatchNorm1d(2048 * 7 * 7)
        # self.layer4_v_1 = nn.BatchNorm1d(2048 * 7 * 7)
        # self.layer4_v_2 = nn.BatchNorm1d(2048 * 7 * 7)
        
        self.layer4_bn1d_0 = nn.BatchNorm1d(2048, affine=False)
        self.layer4_bn1d_1 = nn.BatchNorm1d(2048, affine=False)
        self.layer4_bn1d_2 = nn.BatchNorm1d(2048, affine=False)
        self.layer4_v_0 = nn.Parameter(torch.ones(1, 2048))
        self.layer4_v_1 = nn.Parameter(torch.ones(1, 2048))
        self.layer4_v_2 = nn.Parameter(torch.ones(1, 2048))
        torch.nn.init.normal_(self.layer4_v_0, mean=0.0, std=1)
        torch.nn.init.normal_(self.layer4_v_1, mean=0.0, std=1)
        torch.nn.init.normal_(self.layer4_v_2, mean=0.0, std=1)
        
        # torch.nn.init.normal_(self.layer1_v, mean=0.0, std=1)
        # torch.nn.init.normal_(self.layer2_v, mean=0.0, std=1)
        # torch.nn.init.normal_(self.layer3_v, mean=0.0, std=1)
        # torch.nn.init.normal_(self.layer4_v, mean=0.0, std=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = Identity()
        # normalize output features
        self.l2norm = normalize

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        

    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    # def forward(self, x):
    #     #print(x.shape)
    #     x = self.padding(x)

    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)
    #     #print(x.shape, x.shape[1] * x.shape[2]*x.shape[3])
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)


    #     if self.eval_mode:
    #         return x

    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #    # 0/0
    #     x = self.fc(x)
    #     return x


    

    def forward(self,x):
        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        merged_features = 0 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        # x, features = unfold_layers(self.layer1, x, self.layer1_v)
        # merged_features += features
        # x, features = unfold_layers(self.layer2, x, self.layer2_v)
        # merged_features += features
        
        #x, features = unfold_layers(self.layer3, x, [self.layer3_v_0, self.layer3_v_1, self.layer3_v_2,self.layer3_v_3, self.layer3_v_4, self.layer3_v_5],self.layer4[0].downsample)
        #merged_features += features

        _, features = unfold_layers(self.layer4, x, [self.layer4_v_0, self.layer4_v_1, self.layer4_v_2], None, [self.layer4_bn1d_0, self.layer4_bn1d_1, self.layer4_bn1d_2])
        merged_features += features
        #x = self.avgpool(merged_features.view(x.size(0), 2048, 7, 7))
        x = merged_features
        x = torch.flatten(x, 1)
        return x 



def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

