import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
import torch

enc = {"A": 0, "C": 1, "G": 2, "T": 3}

CHROMOSOMES_TRAIN = ['chr4', 'chr10', 'chr6', 'chr13', 'chr19', 'chr16', 'chr18', 'chr17', 'chr2', 'chr3', 'chr5', 'chr9', 'chr1', 'chr12', 'chr7', 'chr15']
CHROMOSOMES_VAL = ['chr11', 'chr14']
CHROMOSOMES_TEST = ['chrX', 'chr8']

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences, imgs_obs_low, imgs_high, imgs_obs_high):
        super().__init__()
        self.sequences = sequences
        self.imgs_obs_low = imgs_obs_low
        self.imgs_high = imgs_high
        self.imgs_obs_high = imgs_obs_high
    def __getitem__(self, index):
        return self.sequences[index], self.imgs_obs_low[index], \
            self.imgs_high[index], self.imgs_obs_high[index]
    def __len__(self):
        return len(self.sequences)


def get_dataloaders(
        filename_sequences="./dataset/sequences.txt",
        filename_sequences_onehot="./dataset/sequences_onehot.npy",
        filename_low_observed="./dataset/low_tensor_observed_upsampled.npy",
        filename_high="./dataset/high_tensor_v2.npy",
        filename_high_observed="./dataset/high_tensor_observed.npy",
        filename_indexes="./dataset/indexes.npz",
        filename_coords="./dataset/coords.txt",
        chromosomes_train=CHROMOSOMES_TRAIN,
        chromosomes_val=CHROMOSOMES_VAL,
        chromosomes_test=CHROMOSOMES_TEST,
        shuffle=True,
        batch_size=2,
        **kwargs,
        ):

    if os.path.isfile(filename_sequences_onehot):
        sequences = np.load(filename_sequences_onehot)
    else:
        sequences = pickle.load(open(filename_sequences, "rb"))
        sequences_array = []
        for sequence in tqdm(sequences, desc="load_sequences"):
            sequence = np.array([s for s in sequence])
            sequence = np.vectorize(enc.__getitem__)(sequence)
            sequence_onehot = np.zeros((len(sequence), 4), np.bool)
            sequence_onehot[np.arange(len(sequence)), sequence] = 1
            sequences_array.append(sequence_onehot.T)
        sequences = np.array(sequences_array)
        # np.save("./dataset/sequences_onehot.npy", sequences)
    

    tensor_low_observed = np.load(filename_low_observed)
    tensor_high = np.load(filename_high)
    tensor_high_observed = np.load(filename_high_observed)

    if filename_indexes and os.path.isfile(filename_indexes):
        indexes = np.load(filename_indexes)
        indexes_train, indexes_val, indexes_test = indexes["train"], indexes["val"], indexes["test"]
    else:
        coords = pickle.load(open(filename_coords, "rb"))
        indexes_train = np.concatenate([
            np.where([c[0] == ch for c in coords])[0] for ch in chromosomes_train])
        indexes_val = np.concatenate([
            np.where([c[0] == ch for c in coords])[0] for ch in chromosomes_val])
        indexes_test = np.concatenate([
            np.where([c[0] == ch for c in coords])[0] for ch in chromosomes_test])

    dataset_train = Dataset(
        sequences[indexes_train],
        tensor_low_observed[indexes_train],
        tensor_high[indexes_train],
        tensor_high_observed[indexes_train],
    )
    dataset_val = Dataset(
        sequences[indexes_val],
        tensor_low_observed[indexes_val],
        tensor_high[indexes_val],
        tensor_high_observed[indexes_val],
    )
    dataset_test = Dataset(
        sequences[indexes_test],
        tensor_low_observed[indexes_test],
        tensor_high[indexes_test],
        tensor_high_observed[indexes_test],
    )

    def collate_fn(data):
        sequences, tensor_low, tensor_high, tensor_high_2 = zip(*data)
        sequences = torch.from_numpy(np.stack(sequences)).type(torch.float32)
        tensor_low = torch.from_numpy(np.stack(tensor_low))
        tensor_high = torch.from_numpy(np.stack(tensor_high))
        tensor_high_2 = torch.from_numpy(np.stack(tensor_high_2))
        return sequences, tensor_low, tensor_high, tensor_high_2

    kwargs.update({"batch_size": batch_size})
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
        collate_fn=collate_fn, shuffle=shuffle, **kwargs)
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
        collate_fn=collate_fn, **kwargs)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
        collate_fn=collate_fn, **kwargs)

    return dataloader_train, dataloader_val, dataloader_test


if __name__ == "__main__":
    dataloader_train, dataloader_val, dataloader_test = get_dataloaders()
    
    for b in tqdm(dataloader_val):
        seq, t_low, t_high, t_high_2 = b
        