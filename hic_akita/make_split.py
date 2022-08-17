import os
import pickle

import numpy as np


def make_chromosome_based_split(
        filename="./dataset/coords.txt",
        num_train=16,
        num_val=2,
        num_test=2,
        filename_out="./dataset/indexes.npz",
):
    coords = pickle.load(open(filename, "rb"))
    chromosomes_unique = np.unique([c[0] for c in coords]).tolist()
    np.random.shuffle(chromosomes_unique)
    chromosomes_train = chromosomes_unique[:num_train]
    chromosomes_val = chromosomes_unique[num_train:num_train+num_val]
    chromosomes_test = chromosomes_unique[num_train +
                                          num_val:num_train+num_val+num_test]
    indexes_train, indexes_val, indexes_test = [], [], []
    for ch in chromosomes_train:
        indexes_train.append(np.where([c[0] == ch for c in coords])[0])
    for ch in chromosomes_val:
        indexes_val.append(np.where([c[0] == ch for c in coords])[0])
    for ch in chromosomes_test:
        indexes_test.append(np.where([c[0] == ch for c in coords])[0])
    indexes_train, indexes_val, indexes_test = \
        np.concatenate(indexes_train), np.concatenate(
            indexes_val), np.concatenate(indexes_test)
    print("Train:", len(indexes_train), chromosomes_train)
    print("Val:", len(indexes_val), chromosomes_val)
    print("Test:", len(indexes_test), chromosomes_test)
    if filename_out is not None:
        np.savez(filename_out,
                 train=indexes_train,
                 val=indexes_val,
                 test=indexes_test,
                 )
    return indexes_train, indexes_val, indexes_test


if __name__ == "__main__":
    np.random.seed(0)
    make_chromosome_based_split()
