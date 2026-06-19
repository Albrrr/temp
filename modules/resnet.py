import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), inplace=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, if_BN=True):
        super().__init__()
        self.if_BN = if_BN
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes) if if_BN else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if if_BN else nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.leaky_relu(out, inplace=True)

class ResNet(nn.Module):
    def __init__(self, nclasses, aux=False, block=BasicBlock, layers=(3, 4, 6, 3)):
        super().__init__()
        self.aux = aux
        self.inplanes = 128

        self.conv1 = BasicConv2d(5, 64,  kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(128, 128, kernel_size=3, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.conv_1 = BasicConv2d(640, 256, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(256, 128, kernel_size=3, padding=1)
        self.semantic_output = nn.Conv2d(128, nclasses, 1)

        if self.aux:
            self.aux_head1 = nn.Conv2d(128, nclasses, 1)
            self.aux_head2 = nn.Conv2d(128, nclasses, 1)
            self.aux_head3 = nn.Conv2d(128, nclasses, 1)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion),)
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, only_feat=False, return_feat=False):
        x = self.conv3(self.conv2(self.conv1(x)))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        res2 = F.interpolate(x2, size=x.shape[2:], mode='bilinear', align_corners=True)
        res3 = F.interpolate(x3, size=x.shape[2:], mode='bilinear', align_corners=True)
        res4 = F.interpolate(x4, size=x.shape[2:], mode='bilinear', align_corners=True)

        feat = self.conv_2(self.conv_1(torch.cat([x, x1, res2, res3, res4], dim=1)))

        if only_feat:
            return feat

        pred = F.softmax(self.semantic_output(feat), dim=1)
        
        if self.aux:
            aux_outs = [
                F.softmax(self.aux_head1(res2), dim=1),
                F.softmax(self.aux_head2(res3), dim=1),
                F.softmax(self.aux_head3(res4), dim=1),
            ]
            if return_feat:
                return pred, aux_outs, feat
            return pred, aux_outs

        if return_feat:
            return pred, feat
            
        return pred

def ResNet10(nclasses, aux=False):
    return ResNet(nclasses, aux, layers=(1, 1, 1, 1))

def ResNet18(nclasses, aux=False):
    return ResNet(nclasses, aux, layers=(2, 2, 2, 2))

def ResNet34(nclasses, aux=False):
    return ResNet(nclasses, aux, layers=(3, 4, 6, 3))