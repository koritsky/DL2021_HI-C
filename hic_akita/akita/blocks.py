import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *

try:
    from torch_geometric.nn.conv import GCNConv, GMMConv, TransformerConv
except:
    pass


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size=1,
                 stride=1,
                 dilation=1,
                 padding=None,
                 pool_size=1,
                 pool_stride=None,
                 pool_padding=0,
                 batch_norm=False,
                 # bn_momentum=0.99,
                 bn_momentum=0.1,
                 dropout=0,
                 residual=False,
                 ):
        super().__init__()
        if padding is None:
            padding = (kernel_size + (kernel_size-1)*(dilation-1)) // 2
        self.conv = nn.Conv1d(in_channels, filters,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=False)
        if batch_norm:
            self.bn = nn.BatchNorm1d(filters, momentum=bn_momentum)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.activation = nn.ReLU()
        if pool_size > 1:
            self.pool = nn.MaxPool1d(kernel_size=pool_size,
                                     stride=pool_stride, padding=pool_padding)

    def forward(self, inputs):
        x = self.activation(inputs)
        x = self.conv(x)
        if hasattr(self, "bn"):
            x = self.bn(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if self.residual:
            x += inputs
        if hasattr(self, "pool"):
            x = self.pool(x)
        return x


class ConvTower(nn.Module):
    def __init__(self, in_channels, filters_init,
                 filters_mult=1, repeat=1, **kwargs):
        super().__init__()
        channels = [in_channels] + [int(np.round(filters_init * filters_mult**i))
                                    for i in range(repeat)]
        self.layers = nn.Sequential(*[ConvBlock(channels[i], channels[i+1], **kwargs)
                                      for i in range(repeat)])

    def forward(self, x):
        return self.layers(x)


class DilatedResidual(nn.Module):
    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size=3,
                 rate_mult=2,
                 dropout=0,
                 repeat=1,
                 round=False,
                 **kwargs,
                 ):
        super().__init__()
        dilation_rate = 1.
        layers = []
        for ri in range(repeat):
            layers.append(nn.Sequential(
                ConvBlock(in_channels,
                          filters=filters,
                          kernel_size=kernel_size,
                          dilation=int(np.round(dilation_rate)),
                          **kwargs),
                ConvBlock(filters, in_channels,
                          kernel_size=1,
                          dropout=dropout, **kwargs),
            ))
            dilation_rate *= rate_mult
            if round:
                dilation_rate = np.round(dilation_rate)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            output = layer(x)
            x = x + output
        return x


class ConvBlock2d(nn.Module):
    def __init__(self,
                 in_channels,
                 filters=128,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 dropout=0,
                 pool_size=1,
                 pool_stride=None,
                 pool_padding=0,
                 batch_norm=False,
                 # bn_momentum=0.99,
                 bn_momentum=0.1,
                 symmetric=False,
                 ):
        super().__init__()
        self.activation = nn.ReLU()
        if padding is None:
            padding = (kernel_size + (kernel_size-1)*(dilation-1)) // 2
        self.conv = nn.Conv2d(in_channels, out_channels=filters,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        if batch_norm:
            self.bn = nn.BatchNorm2d(filters, momentum=bn_momentum)
        if dropout:
            self.dropout = nn.Dropout2d(dropout)
        if pool_size > 1:
            self.pool = nn.MaxPool2d(kernel_size=pool_size,
                                     stride=pool_stride, padding=pool_padding)
        if symmetric:
            self.symmetrize = Symmetrize2d()

    def forward(self, x):
        x = self.activation(x)
        x = self.conv(x)
        if hasattr(self, "bn"):
            x = self.bn(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if hasattr(self, "pool"):
            x = self.pool(x)
        if hasattr(self, "symmetrize"):
            x = self.symmetrize(x)
        return x


class DilatedResidual2d(nn.Module):
    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size=3,
                 rate_mult=2,
                 dropout=0,
                 repeat=1,
                 symmetric=True,
                 **kwargs,
                 ):
        super().__init__()
        dilation_rate = 1.
        layers = []
        for ri in range(repeat):
            layers.append(nn.Sequential(
                ConvBlock2d(in_channels, filters,
                            kernel_size=kernel_size,
                            dilation=int(np.round(dilation_rate)),
                            **kwargs),
                ConvBlock2d(filters, in_channels, dropout=dropout, **kwargs)
            ))
            if symmetric:
                layers[-1].add_module("symmetric", Symmetrize2d())
            dilation_rate *= rate_mult
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            output = layer(x)
            x = x + output
        return x


class Final(nn.Module):
    def __init__(self,
                 in_channels,
                 units,
                 flatten=False,
                 **kwargs,
                 ):
        super().__init__()
        self.flatten = flatten
        self.dense = nn.Linear(in_channels, units, bias=True)

    def forward(self, x):
        if self.flatten:
            x = self.dense(x.view(-1, x.shape[1]*x.shape[2]))
        else:
            x.transpose_(1, 2)
            x = self.dense(x)
            x.transpose_(1, 2)
        return x


class GraphBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 filters,
                 pos_size,
                 kernel_size,
                 repeat=10,
                 batch_norm=False,
                 bn_momentum=1.-0.9265,
                 dropout=0.1,
                 ):
        super().__init__()
        g_layers = []
        layers = []
        for ri in range(repeat):
            g_layers.append(
                GMMConv(in_channels, filters, pos_size, kernel_size))
            ls = []
            if batch_norm:
                ls.append(nn.BatchNorm1d(filters, momentum=bn_momentum))
            ls.append(nn.ReLU(inplace=True))
            ls.append(ConvBlock(filters, in_channels, dropout=dropout,
                                batch_norm=batch_norm, bn_momentum=bn_momentum))
            layers.append(nn.Sequential(*ls))
        self.g_layers = nn.ModuleList(g_layers)
        self.layers = nn.ModuleList(layers)

    def forward(self, x, inds, emb_unroll):
        b, l, seq_len = x.shape
        for repeat in range(len(self.g_layers)):
            output = self.g_layers[repeat](x.transpose(1, 2).reshape(-1, l),
                                           edge_index=inds, edge_attr=emb_unroll).reshape(b, seq_len, -1).transpose(1, 2)
            output = self.layers[repeat](output)
            x = x + output
        return x


class GraphTransformerBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 heads=1,
                 edge_dim=None,
                 repeat=10,
                 dropout_1=0.1,
                 dropout_2=0.1,
                 batch_norm=False,
                 bn_momentum=1.-0.9265,
                 **kwargs,
                 ):
        super().__init__()
        t_layers = []
        layers = []
        for ri in range(repeat):
            t_layers.append(TransformerConv(in_channels, hidden_dim, heads=heads,
                                            edge_dim=edge_dim, dropout=dropout_1, **kwargs))
            ls = []
            if batch_norm:
                ls.append(nn.BatchNorm1d(
                    heads*hidden_dim, momentum=bn_momentum))
            ls.append(nn.ReLU(inplace=True))
            ls.append(ConvBlock(heads*hidden_dim, in_channels, dropout=dropout_2,
                                batch_norm=batch_norm, bn_momentum=bn_momentum))
            layers.append(nn.Sequential(*ls))
        self.t_layers = nn.ModuleList(t_layers)
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_attr=None):
        b, l, seq_len = x.shape
        for repeat in range(len(self.t_layers)):
            output = self.t_layers[repeat](x.transpose(1, 2).reshape(-1, l),
                                           edge_index=edge_index, edge_attr=edge_attr).reshape(b, seq_len, -1).transpose(1, 2)
            output = self.layers[repeat](output)
            x = x + output
        return x
