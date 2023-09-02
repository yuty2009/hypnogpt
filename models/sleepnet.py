# -*- coding:utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from functools import reduce
from operator import __add__


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)
    

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


if __name__ == '__main__':

    x = torch.randn((20, 1, 3000, 1))
    model = DeepSleepNet(5, 3000)
    # model = TinySleepNet(5, 3000)
    print(model)
    y = model(x)
    print(y.shape)