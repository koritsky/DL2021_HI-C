import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OneToTwo(nn.Module):
    def __init__(self, operation="mean"):
        super().__init__()
        self.operation = operation.lower()
        valid_operations = ["concat", "mean", "max", "multiply", "multiply1"]
        assert self.operation in valid_operations
    
    def forward(self, oned):
        _, features, seq_len = oned.shape

        twod1 = torch.tile(oned, [1, 1, seq_len])
        twod1 = twod1.view(-1, features, seq_len, seq_len)
        twod2 = twod1.transpose(2, 3)

        if self.operation == "concat":
            twod = torch.cat([twod1, twod2], dim=-1)
        elif self.operation == "multiply":
            twod = twod1*twod2
        elif self.operation == "multiply1":
            twod = (twod1+1)*(twod2+1) - 1
        else:
            twod1.unsqueeze_(-1)
            twod2.unsqueeze_(-1)
            twod = torch.cat([twod1, twod2], dim=-1)
            if self.operation == "mean":
                twod = twod.mean(dim=-1)
            elif self.operation == "max":
                twod = torch.cat([twod1, twod2]).max(dim=-1)
        return twod


class ConcatDist2d(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        input_shape = inputs.shape
        batch_size, seq_len = input_shape[0], input_shape[2]
        
        pos = torch.arange(seq_len, device=inputs.device).unsqueeze(-1)
        matrix_repr1 = torch.tile(pos, [1, seq_len])
        matrix_repr2 = matrix_repr1.transpose(1, 0)
        dist = (matrix_repr1 - matrix_repr2).abs().type(torch.float32)
        dist = dist.unsqueeze(0).unsqueeze(1)
        dist = torch.tile(dist, [batch_size, 1, 1, 1])
        return torch.cat([inputs, dist], dim=1)


class ConcatDist2dEmbedding(nn.Module):
    def __init__(self, num=512, out=16):
        super().__init__()
        self.emb = nn.Embedding(num, out)
    def forward(self, inputs):
        input_shape = inputs.shape
        batch_size, seq_len = input_shape[0], input_shape[2]
        
        pos = torch.arange(seq_len, device=inputs.device).unsqueeze(-1)
        matrix_repr1 = torch.tile(pos, [1, seq_len])
        matrix_repr2 = matrix_repr1.transpose(1, 0)
        dist = (matrix_repr1 - matrix_repr2).abs()
        emb = self.emb(dist.view(1, -1)).view(1, -1, seq_len, seq_len)
        emb = torch.tile(emb, [batch_size, 1, 1, 1])
        return torch.cat([inputs, emb], dim=1)



class Symmetrize2d(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x_t = x.transpose(2, 3)
        return (x+x_t)/2

class Cropping2d(nn.Module):
    def __init__(self, cropping=32):
        super().__init__()
        self.cropping = cropping
    def forward(self, x):
        _, _, s1, s2 = x.shape
        return x[:, :, 
            self.cropping:s1-self.cropping, 
            self.cropping:s2-self.cropping]


class UpperTri(nn.Module):
    def __init__(self, diagonal_offset=2):
        super().__init__()
        self.diagonal_offset = diagonal_offset

    def forward(self, inputs):
        b, output_dim, seq_len, _ = inputs.shape

        triu_tup = np.triu_indices(seq_len, self.diagonal_offset)
        triu_index = torch.from_numpy(triu_tup[0] + seq_len*triu_tup[1]).to(inputs.device)
        triu_index = triu_index.unsqueeze(0).unsqueeze(1).repeat(b, output_dim, 1)
        unroll_repr = inputs.reshape(-1, output_dim, seq_len**2)
        return unroll_repr.gather(-1, index=triu_index)


class StochasticReverseComplement(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, seq_1hot):
        if self.training:
            if np.random.random() > 0.5:
                # bs, num, seq_len = seq_1hot.shape
                # inds = torch.arange(num, 0, -1, device=seq_1hot.device)
                # inds = inds.unsqueeze(0).unsqueeze(-1).repeat(bs, 1, seq_len)
                # rc_seq_1hot = seq_1hot.gather(-1, inds).flip([-1])
                rc_seq_1hot = torch.flip(seq_1hot, dims=(1, 2))
                return rc_seq_1hot, True
            else:
                return seq_1hot, False
        else:
            return seq_1hot, False


# def shift_sequence(seq, shift, pad_value=0.25):
def shift_sequence(seq, shift, pad_value=0):
    pad = pad_value * torch.ones_like(seq[:, :, 0:np.abs(shift)])
    def _shift_right(_seq):
        sliced_seq = _seq[:, :, :-shift:]
        return torch.cat([pad, sliced_seq], dim=2)
    def _shift_left(_seq):
        sliced_seq = _seq[:, :, -shift:]
        return torch.cat([sliced_seq, pad], dim=2)
    if shift > 0:
        sseq = _shift_right(seq)
    else:
        sseq = _shift_left(seq)
    return sseq


class StochasticShift(nn.Module):
    def __init__(self, shift_max=0, symmetric=True):
        super().__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        if self.symmetric:
            self.augment_shifts = torch.arange(-self.shift_max, self.shift_max+1)
        else:
            self.augment_shifts = torch.arange(0, self.shift_max+1)
        
    def forward(self, seq_1hot):
        if self.training:
            shift_i = np.random.choice(self.augment_shifts.numpy())
            if shift_i == 0:
                return seq_1hot
            else:
                return shift_sequence(seq_1hot, self.augment_shifts[shift_i])
        else:
            return seq_1hot


class SwitchReverse(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, reverse):
        xd = len(x.shape)
        if xd == 3:
            rev_axes = [2]
        elif xd == 4:
            rev_axes = [2, 3]
        if reverse:
            return x.flip(rev_axes)
        else:
            return x

class SwitchReverseTriu(nn.Module):
    def __init__(self, diagonal_offset):
        super().__init__()
        self.diagonal_offset = diagonal_offset

    def forward(self, x, reverse):
        if reverse:
            bs, num, ut_len = x.shape
            seq_len = int(np.sqrt(2*ut_len + 0.25) - 0.5)
            seq_len += self.diagonal_offset
            ut_indexes = np.triu_indices(seq_len, self.diagonal_offset)
            mat_ut_indexes = np.zeros(shape=(seq_len, seq_len), dtype=np.int32)
            mat_ut_indexes[ut_indexes] = np.arange(ut_len)
            mask_ut = np.zeros(shape=(seq_len, seq_len), dtype=np.bool)
            mask_ut[ut_indexes] = True
            mask_ld = ~mask_ut
            mat_indexes = mat_ut_indexes + np.multiply(mask_ld, mat_ut_indexes.T)
            mat_rc_indexes = mat_indexes[::-1, ::-1]
            rc_ut_order = torch.from_numpy(mat_rc_indexes[ut_indexes]).to(x.device).type(torch.int64)
            rc_ut_order = rc_ut_order.unsqueeze(0).unsqueeze(1).repeat(bs, num, 1)
            return x.gather(-1, rc_ut_order)
        else:
            return x
        

class PosEmbedding(nn.Module):
    def __init__(self, num=512, out=8):
        super().__init__()
        self.emb = nn.Embedding(num, out)
    def forward(self, x):
        bs, n, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(-1)
        matrix_repr1 = torch.tile(pos, [1, seq_len])
        matrix_repr2 = matrix_repr1.transpose(1, 0)
        dist = (matrix_repr1 - matrix_repr2).abs()
        emb = self.emb(dist.view(1, -1)).view(1, -1, seq_len, seq_len)
        emb = torch.tile(emb, [bs, 1, 1, 1])
        return emb

class PosEmbedding1d(nn.Module):
    def __init__(self, num=512, out=8):
        super().__init__()
        self.emb = nn.Embedding(num, out)
    def forward(self, x):
        bs, n, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        emb = self.emb(pos)
        emb = torch.tile(emb.view(1, -1, seq_len), [bs, 1, 1])
        return emb

def pos_encoding(x):
    bs, n, seq_len = x.shape
    pos = torch.arange(seq_len, device=x.device).unsqueeze(-1)
    matrix_repr1 = torch.tile(pos, [1, seq_len])
    matrix_repr2 = matrix_repr1.transpose(1, 0)
    dist = (matrix_repr1 - matrix_repr2).abs()
    dist = torch.tile(dist, [bs, 1, 1, 1])
    return dist

def get_indexes_embs(x, emb):
    b, l, seq_len = x.shape
    triu_tup = np.triu_indices(seq_len)
    triu_index = torch.from_numpy(triu_tup[0] + seq_len*triu_tup[1]).to(x.device)
    triu_index = triu_index.unsqueeze(0).unsqueeze(1).repeat(b, emb.shape[1], 1)
    emb_unroll = emb.reshape(-1, emb.shape[1], seq_len**2).gather(-1, index=triu_index)
    emb_unroll = emb_unroll.transpose(1, 2).reshape(-1, emb_unroll.shape[1])
    triu_tup = np.stack([triu_tup[0], triu_tup[1]])
    inds = torch.from_numpy(np.concatenate([i*seq_len+triu_tup for i in range(b)], axis=1)).to(x.device)
    return inds, emb_unroll


class OneToTwo2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_dim, in_dim, in_dim))
        self.bias = torch.nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        bs, n, seq_len = x.shape
        x1 = x.unsqueeze(1).transpose(2, 3)
        out = torch.matmul(x1, self.weight)
        out = torch.matmul(out, x.unsqueeze(1))
        return out + self.bias.view(1, -1, 1, 1)


def get_2d_from_upper_tri_only(x):
    b, n, s, _ = x.shape
    mask_tri = (torch.tril(torch.ones(s, s)) == 1).unsqueeze(0).unsqueeze(1).repeat(b, n, 1, 1)
    mask_diag = torch.eye(s, dtype=torch.bool).unsqueeze(0).unsqueeze(1).repeat(b, n, 1, 1)
    x_tri = torch.zeros_like(x)
    x_tri[mask_tri] = x[mask_tri]
    out = x_tri + x_tri.transpose(2, 3)
    out[mask_diag] = x[mask_diag]
    return out
    
class UpperTri2dTo2d(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x):
        return get_2d_from_upper_tri_only(x)