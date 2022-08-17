import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

enc = {"A": 0, "C": 1, "G": 2, "T": 3}

CHROMOSOMES_TRAIN = ['chr4', 'chr10', 'chr6', 'chr13', 'chr19', 'chr16', 'chr18', 'chr17', 'chr2', 'chr3', 'chr5', 'chr9', 'chr1', 'chr12', 'chr7', 'chr15']
CHROMOSOMES_VAL = ['chr11', 'chr14']
CHROMOSOMES_TEST = ['chrX', 'chr8']

def convert_data(
        filename_sequences="./dataset/sequences.txt",
        filename_images="./dataset/high_tensor_v2.npy",
        filename_coords="./dataset/coords.txt",
        filename_out="./dataset/data.npz",
        ):
    images = np.load(filename_images)
    sequences = pickle.load(open(filename_sequences, "rb"))
    coords = pickle.load(open(filename_coords, "rb"))
    
    indexes_train = np.concatenate([
        np.where([c[0] == ch for c in coords])[0] for ch in CHROMOSOMES_TRAIN])
    indexes_val = np.concatenate([
        np.where([c[0] == ch for c in coords])[0] for ch in CHROMOSOMES_VAL])
    indexes_test = np.concatenate([
        np.where([c[0] == ch for c in coords])[0] for ch in CHROMOSOMES_TEST])

    print("train/val/test:", len(indexes_train), len(indexes_val), len(indexes_test))

    sequences_array = []
    for sequence in tqdm(sequences, desc="sequences"):
        sequence = np.array([s for s in sequence])
        sequence = np.vectorize(enc.__getitem__)(sequence)
        sequence_onehot = np.zeros((len(sequence), 4), np.bool)
        sequence_onehot[np.arange(len(sequence)), sequence] = 1
        sequences_array.append(sequence_onehot.T)
    sequences = np.array(sequences_array)

    if filename_out:
        np.savez(
            filename_out,
            sequences=sequences,
            targets=images,
            indexes_train=indexes_train,
            indexes_val=indexes_val,
            indexes_test=indexes_test,
        )
    return 


if __name__ == "__main__":
    convert_data()