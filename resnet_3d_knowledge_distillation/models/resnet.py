import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)

def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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

class ResNet3D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 num_class = 5,
                 no_cuda=False):

        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet3D, self).__init__()

        # 3D conv net
        self.conv1 = nn.Conv3d(53, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        # self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 64*2, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 128*2, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 256*2, layers[3], shortcut_type, stride=1, dilation=4)
        
        self.fea_dim = 256*2 * block.expansion

        self.relu_cnn = nn.ReLU(inplace=True)
        self.relu_concat = nn.ReLU(inplace=True)
        

        self.fc_feat = nn.Sequential(nn.Linear(82, self.fea_dim, bias=True))
        self.relu_feat = nn.ReLU(inplace=True)
        
        self.fc_fnc = nn.Sequential(nn.Linear(1378, self.fea_dim, bias=True))
        self.relu_fnc = nn.ReLU(inplace=True)
        
        self.fc_deg = nn.Sequential(nn.Linear(53, self.fea_dim, bias=True))
        self.relu_deg = nn.ReLU(inplace=True) 
        
        self.last_fc1_1 = nn.Sequential(nn.Linear(self.fea_dim * 4, self.fea_dim * 4, bias=True))
        self.bn_fc1_1 = nn.BatchNorm1d(self.fea_dim * 4)
        self.relu_fc1_1 = nn.ReLU(inplace=True)
        
        self.last_fc1_2 = nn.Sequential(nn.Linear(self.fea_dim * 4, self.fea_dim * 4, bias=True))
        self.bn_fc1_2 = nn.BatchNorm1d(self.fea_dim * 4)
        self.relu_fc1_2 = nn.ReLU(inplace=True)
        
        self.last_fc1_3 = nn.Sequential(nn.Linear(self.fea_dim * 4, self.fea_dim * 4, bias=True))
        self.bn_fc1_3 = nn.BatchNorm1d(self.fea_dim * 4)
        self.relu_fc1_3 = nn.ReLU(inplace=True)
        
        self.last_fc1 = nn.Linear(self.fea_dim * 4, 1, bias=True) #nn.Sequential(nn.Linear(self.fea_dim * 3, num_class, bias=True))
        torch.nn.init.normal_(self.last_fc1.weight, std=0.02)
        self.high_dropout = nn.Dropout(p=0.4)
        
        
        self.last_fc2_1 = nn.Sequential(nn.Linear(self.fea_dim * 4, self.fea_dim * 4, bias=True))
        self.bn_fc2_1 = nn.BatchNorm1d(self.fea_dim * 4)
        self.relu_fc2_1 = nn.ReLU(inplace=True)
        
        self.last_fc2_2 = nn.Sequential(nn.Linear(self.fea_dim * 4, self.fea_dim * 4, bias=True))
        self.bn_fc2_2 = nn.BatchNorm1d(self.fea_dim * 4)
        self.relu_fc2_2 = nn.ReLU(inplace=True)
        
        self.last_fc2_3 = nn.Sequential(nn.Linear(self.fea_dim * 4, self.fea_dim * 4, bias=True))
        self.bn_fc2_3 = nn.BatchNorm1d(self.fea_dim * 4)
        self.relu_fc2_3 = nn.ReLU(inplace=True)
        
        self.last_fc2 = nn.Linear(self.fea_dim * 4, 1, bias=True) #nn.Sequential(nn.Linear(self.fea_dim * 3, num_class, bias=True))
        torch.nn.init.normal_(self.last_fc2.weight, std=0.02)
        

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x, x_feat, x_fnc, x_deg):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view((-1, self.fea_dim))
        x = self.relu_cnn(x)
        
        x_feat = self.fc_feat(x_feat)
        x_feat = self.relu_feat(x_feat)
        
        x_fnc = self.fc_fnc(x_fnc)
        x_fnc = self.relu_fnc(x_fnc)
        
        x_deg = self.fc_deg(x_deg)
        x_deg = self.relu_deg(x_deg)
        
        x_cat = torch.cat([x, x_feat, x_fnc, x_deg], dim=1)
       # x = self.relu_concat(x)
        
        res = self.last_fc1_1(x_cat)
        res = self.bn_fc1_1(res)
        out1 = res + x_cat
        out1 = self.relu_fc1_1(out1)
        
        res = self.last_fc1_2(out1)
        res = self.bn_fc1_2(res)
        out1 = res + out1
        out1 = self.relu_fc1_2(out1)   
        
        res = self.last_fc1_3(out1)
        res = self.bn_fc1_3(res)
        out1 = res + out1
        out1 = self.relu_fc1_3(out1) 
        
        out1 = torch.mean(
            torch.stack(
                [self.last_fc1(self.high_dropout(out1)) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )
        
        res = self.last_fc2_1(x_cat)
        res = self.bn_fc2_1(res)
        out2 = res + x_cat
        out2 = self.relu_fc2_1(out2)
        
        res = self.last_fc2_2(out2)
        res = self.bn_fc2_2(res)
        out2 = res + out2
        out2 = self.relu_fc2_2(out2)   
        
        res = self.last_fc2_3(out2)
        res = self.bn_fc2_3(res)
        out2 = res + out2
        out2 = self.relu_fc2_3(out2) 
        
        out2 = self.last_fc2(out2)
        return out1, out2


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(BasicBlock, [1, 1, 1, 1],**kwargs)
    return model

def resnet3d_10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
