import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from .layers import UpperTri, Cropping2d


TARGET_MIN = -2.
TARGET_MAX = 2.
TARGET_MEAN = -0.11719574
TARGET_STD = 0.481647


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets):
        super().__init__()
        self.sequences = sequences
        self.targets = targets
    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]
    def __len__(self):
        return len(self.sequences)


def get_dataloaders(
        filename="./dataset/data.npz",
        diagonal_offset=2,
        target_crop=22,
        shuffle=True,
        batch_size=2,
        num_workers=2,
        **kwargs,
        ):
    
    data = np.load(filename)
    sequences = data["sequences"]
    targets = data["targets"]
    indexes_train, indexes_val, indexes_test = \
        data["indexes_train"], data["indexes_val"], data["indexes_test"]

    dataset_train = Dataset(
        sequences[indexes_train] if isinstance(sequences, np.ndarray) else [sequences[i] for i in indexes_train],
        targets[indexes_train],)
    dataset_val = Dataset(
        sequences[indexes_val] if isinstance(sequences, np.ndarray) else [sequences[i] for i in indexes_val],
        targets[indexes_val],)
    dataset_test = Dataset(
        sequences[indexes_test] if isinstance(sequences, np.ndarray) else [sequences[i] for i in indexes_test],
        targets[indexes_test])

    upper_tri = UpperTri(diagonal_offset=diagonal_offset)
    if target_crop > 0:
        cropping = Cropping2d(target_crop)

    def collate_fn(data):
        sequences, targets = zip(*data)
        sequences = torch.from_numpy(np.stack(sequences)).type(torch.float32)
        targets = torch.from_numpy(np.stack(targets))
        if target_crop > 0:
            targets = cropping(targets.unsqueeze(1))
        else:
            targets = targets.unsqueeze(1)
        targets_tri = upper_tri(targets)
        targets_tri.squeeze_(1)
        targets.squeeze_(1)
        return sequences, targets, targets_tri

    kwargs.update({"batch_size": batch_size, "num_workers": num_workers})
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
        collate_fn=collate_fn, shuffle=shuffle, **kwargs)
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
        collate_fn=collate_fn, **kwargs)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
        collate_fn=collate_fn, **kwargs)

    return dataloader_train, dataloader_val, dataloader_test


def from_upper_triu(vector_repr, matrix_len, num_diags):
    z = np.zeros((matrix_len,matrix_len))
    triu_tup = np.triu_indices(matrix_len,num_diags)
    z[triu_tup] = vector_repr
    return z + z.T


def convert_gr_to_rb(image):
    color_red = [1, 0, 0]
    color_blue = [0, 0, 1]
    b1 = image >= 0
    b2 = image < 0
    image = (image + 2) / 4
    image3 = torch.zeros((3, image.shape[0], image.shape[1]), device=image.device,
        dtype=image.dtype)
    for i in range(3):
        image3[i][b1] = 1. - (1. - color_red[i])*2*(image[b1] - 0.5)
        image3[i][b2] = 1. - (1. - color_blue[i])*2*(0.5 - image[b2])
    return image3
