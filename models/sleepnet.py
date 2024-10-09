# -*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
    

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x
    

class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(-2)
    input_cols = input.size(-1)
    filter_rows = weight.size(-2)
    filter_cols = weight.size(-1)
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    out_cols = (input_cols + stride[1] - 1) // stride[1]
    padding_rows = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_cols - 1) * stride[1] + (filter_cols - 1) * dilation[1] + 1 - input_cols)
    cols_odd = (padding_cols % 2 != 0)

    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input.clone(), weight, bias, stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)

class Conv2dSamePadding(_ConvNd): 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dSamePadding, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
    

class Conv2dBnReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        ConvLayer = Conv2dSamePadding(
            in_channels, out_channels, kernel_size, stride, bias=False, **kwargs
        )
        super(Conv2dBnReLU, self).__init__(
            ConvLayer,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Inception2d(nn.Module):
    def conv_block(self, in_channels, out_channels, kernel_size, stride, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(pool, 1), stride=(pool, 1))
        )
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pool=2):
        super(Inception2d, self).__init__()
        num_branches = len(kernel_size)
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * num_branches
        assert out_channels % num_branches == 0, 'out_channels must be divisible by num_branches'
        branch_channels = out_channels // num_branches
        branches = []
        for i in range(num_branches):
            branch = self.conv_block(
                in_channels, branch_channels, kernel_size[i], stride[i], pool=pool
            )
            branches.append(branch)
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        x = [branch(x) for branch in self.branches]
        x = torch.cat(x, 1)
        return x
    

def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)

        self.convs = nn.ModuleList([])
        for k, in_ch, out_ch in zip(kernel_size, in_splits, out_splits):
            conv_groups = out_ch if depthwise else 1
            self.convs.append(nn.Sequential(
                Conv2dSamePadding(
                    in_ch, out_ch, k, stride=stride,
                    padding=0, dilation=dilation, groups=conv_groups, **kwargs),
            ))
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.convs)]
        x = torch.cat(x_out, 1)
        return x
    

class SKConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1, depthwise=False, 
                 r=16, L=32, **kwargs):
        super(SKConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        self.M = len(kernel_size)
        self.features = out_channels
        d = max(int(out_channels/r), L)

        self.convs = nn.ModuleList([])
        for k in kernel_size:
            conv_groups = out_channels if depthwise else 1
            self.convs.append(nn.Sequential(  
                Conv2dSamePadding(
                    in_channels, out_channels, k, stride=stride,
                    padding=0, dilation=dilation, groups=conv_groups, **kwargs
                ),
                # nn.BatchNorm2d(out_ch),
                # nn.ReLU(inplace=False),
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, d, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(d),
            # nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([])
        for i in range(self.M):
            self.fcs.append(
                nn.Linear(d, self.features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        feats = [conv(x) for conv in self.convs]      
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[-2], feats.shape[-1])
        
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        feats_Z = feats_Z.flatten(1)

        attention_vectors = [fc(feats_Z).unsqueeze_(dim=1) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)

        fea_v = (feats * attention_vectors).sum(dim=1)

        return fea_v
    

class SKUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, kernel_size=3,
                 stride=1, dilation=1, r=16, L=32, **kwargs):
        super(SKUnit, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            )
        
        self.conv2_sk = SKConv2d(
            mid_features, mid_features, kernel_size=kernel_size,
            stride=stride, dilation=dilation, r=r, L=L, **kwargs
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
            )
        

        if in_features == out_features: # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)
        
        return self.relu(out + self.shortcut(residual))

        
class DeepSleepNet(nn.Module):
    """
    Reference:
    A. Supratak, H. Dong, C. Wu, and Y. Guo, "DeepSleepNet: A Model for Automatic
    Sleep Stage Scoring Based on Raw Single-Channel EEG," IEEE Trans Neural Syst 
    Rehabil Eng, vol. 25, no. 11, pp. 1998-2008, 2017.
    https://github.com/akaraspt/deepsleepnet
    """
    def __init__(
        self, n_classes, n_timepoints, dropout = 0.5,
        # Conv layers
        n_filters_1 = 64, filter_size_1 = 50, filter_stride_1 = 6,
        n_filters_2 = 64, filter_size_2 = 400, filter_stride_2 = 50,
        pool_size_11 = 8, pool_stride_11 = 8, 
        pool_size_21 = 4, pool_stride_21 = 4,
        n_filters_1x3 = 128, filter_size_1x3 = 8,
        n_filters_2x3 = 128, filter_size_2x3 = 6,
        pool_size_12 = 4, pool_stride_12 = 4, 
        pool_size_22 = 2, pool_stride_22 = 2,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv2dBnReLU(1, n_filters_1, (filter_size_1, 1), (filter_stride_1, 1)),
            nn.MaxPool2d((pool_size_11, 1), (pool_stride_11, 1)),
            nn.Dropout(dropout),
            Conv2dBnReLU(n_filters_1,   n_filters_1x3, (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3, (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3, (filter_size_1x3, 1), stride=1),
            nn.MaxPool2d((pool_size_12, 1), (pool_stride_12, 1)),
        )
        self.conv2 = nn.Sequential(
            Conv2dBnReLU(1, n_filters_2, (filter_size_2, 1), (filter_stride_2, 1)),
            nn.MaxPool2d((pool_size_21, 1), (pool_stride_21, 1)),
            nn.Dropout(dropout),
            Conv2dBnReLU(n_filters_2,   n_filters_2x3, (filter_size_2x3, 1), stride=1),
            Conv2dBnReLU(n_filters_2x3, n_filters_2x3, (filter_size_2x3, 1), stride=1),
            Conv2dBnReLU(n_filters_2x3, n_filters_2x3, (filter_size_2x3, 1), stride=1),
            nn.MaxPool2d((pool_size_22, 1), (pool_stride_22, 1)),
        )
        self.drop1 = nn.Dropout(dropout)

        outlen_conv1 = n_timepoints // filter_stride_1 // pool_stride_11 // pool_stride_12
        outlen_conv2 = n_timepoints // filter_stride_2 // pool_stride_21 // pool_stride_22
        outlen_conv = outlen_conv1*n_filters_1x3 + outlen_conv2*n_filters_2x3

        self.feature_dim = outlen_conv
        self.classifier = nn.Linear(outlen_conv, n_classes) if n_classes > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = x1.view(x1.size(0), -1) # flatten (b, c, t, 1) -> (b, c*t)
        x2 = x2.view(x2.size(0), -1) # flatten (b, c, t, 1) -> (b, c*t)
        x = torch.cat((x1, x2), dim=1) # concat in feature dimention
        x = self.drop1(x)
        x = self.classifier(x)
        return x


class TinySleepNet(nn.Module):
    """
    Reference:
    A. Supratak and Y. Guo, "TinySleepNet: An Efficient Deep Learning Model
    for Sleep Stage Scoring based on Raw Single-Channel EEG," Annu Int Conf
    IEEE Eng Med Biol Soc, vol. 2020, pp. 641-644, Jul 2020.
    https://github.com/akaraspt/tinysleepnet
    """
    def __init__(
        self, n_classes, n_timepoints, dropout = 0.5,
        # Conv layers
        n_filters_1 = 128, filter_size_1 = 50, filter_stride_1 = 6,
        pool_size_1 = 8, pool_stride_1 = 8, 
        n_filters_1x3 = 128, filter_size_1x3 = 8,
        pool_size_2 = 4, pool_stride_2 = 4, 
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv2dBnReLU(1, n_filters_1, (filter_size_1, 1), (filter_stride_1, 1)),
            nn.MaxPool2d((pool_size_1, 1), (pool_stride_1, 1)),
            nn.Dropout(dropout),
            Conv2dBnReLU(n_filters_1,   n_filters_1x3, (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3, (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3, (filter_size_1x3, 1), stride=1),
            nn.MaxPool2d((pool_size_2, 1), (pool_stride_2, 1)),
            nn.Dropout(dropout)
        )

        outlen_conv1 = n_timepoints // filter_stride_1 // pool_stride_1 // pool_stride_2
        outlen_conv = outlen_conv1*n_filters_1x3

        self.feature_dim = outlen_conv
        self.classifier = nn.Linear(outlen_conv, n_classes) if n_classes > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1) # flatten (b, c, t, 1) -> (b, c*t)
        x = self.classifier(x)
        return x


class MSCSleepNet(nn.Module):
    # Multi-scale convolutional network
    def __init__(
        self, n_classes, n_timepoints, dropout = 0.5,
        # Conv layers
        n_filters_1 = 128, filter_size_1 = [(10, 1), (20, 1), (50, 1)], filter_stride_1 = 6,
        pool_size_1 = 8, pool_stride_1 = 8,
        n_filters_1x3 = 128, filter_size_1x3 = 8,
        pool_size_2 = 2, pool_stride_2 = 2,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            # Conv2dBnReLU(1, n_filters_1, (filter_size_1, 1), (filter_stride_1, 1)),
            SKConv2d(1, n_filters_1, kernel_size=filter_size_1, stride=filter_stride_1),
            nn.BatchNorm2d(n_filters_1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((pool_size_1, 1), (pool_stride_1, 1)),
            nn.Dropout(dropout),
            Conv2dBnReLU(n_filters_1,   n_filters_1x3, (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3, (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3, (filter_size_1x3, 1), stride=1),
            nn.MaxPool2d((pool_size_2, 1), (pool_stride_2, 1)),
            nn.Dropout(dropout)
        )

        outlen_conv1 = n_timepoints // filter_stride_1 // pool_stride_1 // pool_stride_2
        outlen_conv = outlen_conv1*n_filters_1x3

        self.feature_dim = outlen_conv
        self.classifier = nn.Linear(outlen_conv, n_classes) if n_classes > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1) # flatten (b, c, t, 1) -> (b, c*t)
        x = self.classifier(x)
        return x


if __name__ == '__main__':

    x = torch.randn((20, 1, 3000, 1))
    # model = DeepSleepNet(5, 3000)
    # model = TinySleepNet(5, 3000)
    # model = ConvSleepNet(5, 3000)
    model = MSCSleepNet(5, 3000)
    print(model)
    y = model(x)
    print(y.shape)
