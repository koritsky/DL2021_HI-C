import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from .layers import *
from .blocks import *

from .utils import TARGET_MIN, TARGET_MAX, TARGET_MEAN, TARGET_STD


class Model(nn.Module):
    def __init__(self,
            diagonal_offset=2,
            target_crop=10,
            augment_rc=True,
            augment_shift=11,
            preds_triu=True,
            # 
            num_conv_tower_1d=11,
            num_targets=1,
            dropout_1=0.4,
            target_size=244,
            target_size_interpolate=200,
            # 
            batch_norm=True,
            bn_momentum=1.-0.9265,
            # 
            **kwargs,
            ):
        super().__init__()
        self.diagonal_offset = diagonal_offset
        self.target_size = target_size
        self.target_crop = target_crop
        self.target_size_interpolate = target_size_interpolate
        
        self.preds_triu = preds_triu
        if augment_rc:
            self.augment_rc = StochasticReverseComplement()
            if self.preds_triu:
                self.rc = SwitchReverseTriu(augment_shift)
            else:
                self.rc = SwitchReverse()
        if augment_shift != 0:
            self.augment_shift = StochasticShift(augment_shift)

        self.trunk = nn.Sequential(
            ConvBlock(4,
                filters=96,
                kernel_size=11,
                pool_size=2,
                pool_stride=2,
                batch_norm=batch_norm,
                bn_momentum=bn_momentum,
            ),
            ConvTower(96,
                filters_init=96,
                filters_mult=1.,
                kernel_size=5,
                pool_size=2,
                pool_stride=2,
                repeat=num_conv_tower_1d,
                batch_norm=batch_norm,
                bn_momentum=bn_momentum,
            ),
            DilatedResidual(96,
                filters=48,
                rate_mult=1.75,
                repeat=8,
                dropout=dropout_1,
                batch_norm=batch_norm,
                bn_momentum=bn_momentum,
            ),
            ConvBlock(96,
                filters=64,
                kernel_size=5,
                batch_norm=batch_norm,
                bn_momentum=bn_momentum,
            )
        )

        self.init_middle(batch_norm=batch_norm, bn_momentum=bn_momentum, **kwargs)
        self.cropping = Cropping2d(target_crop)
        self.totri = UpperTri(diagonal_offset)
        self.final = nn.Sequential(Final(48, num_targets))
        self.init_weights()


    def init_middle(self, *args, **kwargs):
        pass
    def middle(self, *args, **kwargs):
        pass

    @property
    def image_size(self):
        if self.target_size_interpolate:
            return self.target_size_interpolate - 2*self.target_crop
        else:
            return self.target_size - 2*self.target_crop


    def forward(self, x, triu=None):
        if triu is None: triu = self.preds_triu
        if hasattr(self, "augment_rc"):
            x, reverse_bool = self.augment_rc(x)
        if hasattr(self, "augment_shift"):
            x = self.augment_shift(x)
        x = self.trunk(x)
        x.relu_()

        x = self.middle(x)

        b, n, s, _ = x.shape
        x = self.final(x.view(b, n, s*s)).view(b, -1, s, s)
        if self.target_size_interpolate and self.target_size != self.target_size_interpolate:
            x = F.interpolate(x, 
                size=[self.target_size_interpolate, self.target_size_interpolate],
                mode="bilinear")

        x = self.cropping(x)
        if triu:
            x = self.totri(x)
        if hasattr(self, "rc"):
            x = self.rc(x, reverse_bool)

        x = torch.clip(TARGET_MEAN + x.squeeze(1)*TARGET_STD,
            TARGET_MIN, TARGET_MAX)
        return x


    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)



class ModelAkita(Model):
    def init_middle(self, dropout_2=0.1, **kwargs):
        self.head = nn.Sequential(
            OneToTwo(operation="mean"),
            ConcatDist2d(),
            ConvBlock2d(64+1,
                filters=48,
                kernel_size=3,
                **kwargs,
            ),
            Symmetrize2d(),
            DilatedResidual2d(48,
                filters=24,
                kernel_size=3,
                rate_mult=1.75,
                repeat=6,
                dropout=dropout_2,
                **kwargs,
            ),
        )

    def middle(self, x):
        return self.head(x)


class ModelWGraph(Model):
    def init_middle(self, 
            num_hidden=64,
            num_heads=2,
            num_emb=8,
            repeat=4,
            dropout_1=0.3, 
            dropout_2=0.1,
            bn_momentum=1.-0.9265,
            symmetrize=False,
            **kwargs,
            ):
        self.graph_block = GraphTransformerBlock(
            64, num_hidden, num_heads, num_emb, repeat=repeat,
            dropout_1=dropout_1, dropout_2=dropout_2,
            **kwargs,
        )
        self.pos_emb = PosEmbedding(self.target_size, 8)
        if symmetrize:
            self.onetotwo = nn.Sequential(
                OneToTwo2(64, 48),
                Symmetrize2d(),
                nn.BatchNorm2d(48, momentum=bn_momentum),
                nn.ReLU(),
            )
        else:
            self.onetotwo = nn.Sequential(
                OneToTwo2(64, 48),
                nn.BatchNorm2d(48, momentum=bn_momentum),
                nn.ReLU(),
                UpperTri2dTo2d(),
            )

    def middle(self, x):
        emb = self.pos_emb(x)
        inds, emb_unroll = get_indexes_embs(x, emb)
        x = self.graph_block(x, inds, emb_unroll)
        x = self.onetotwo(x)
        return x