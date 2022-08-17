import glob
import os
import pdb
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from Utils import utils as ut


class MouseModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    class MouseDataset(Dataset):
            def __init__(self,
                          tvt,
                          data_path,
                          target_path,
                          indexes_path):

                assert tvt in ['train', 'val', 'test'], 'tvt can be `train`, `val` or `test`'

                self.tvt = tvt
                self.idxs = np.load(indexes_path)[self.tvt]

                '''
                We use the following chromosomes split:
                Train: 1912 ['chr4', 'chr10', 'chr6', 'chr13', 'chr19', 'chr16', 'chr18', 'chr17', 'chr2', 'chr3', 'chr5', 'chr9', 'chr1', 'chr12', 'chr7', 'chr15']
                Val: 231 ['chr11', 'chr14']
                Test: 256 ['chrX', 'chr8']
                '''

                self.data   = np.load(data_path)[self.idxs]
                self.target = np.load(target_path)[self.idxs]
                
                #add channels dimension for working with native pytorch
                self.data  = np.expand_dims(self.data, axis=1)
                self.target  = np.expand_dims(self.target, axis=1)

            def __len__(self):
                return self.data.shape[0]

            def __getitem__(self, idx):
                return self.data[idx], self.target[idx]

    def setup(self,
              stage=None,
              data_path = 'Data/Mouse/augmented/low_tensor_observed_upsampled.npy',
              target_path = 'Data/Mouse/augmented/high_tensor_observed.npy',
              indexes_path = 'Data/Mouse/augmented/indexes.npz'
              # data_path = 'Data/Mouse/low_tensor_observed_upsampled.npy',
              # target_path = 'Data/Mouse/high_tensor_observed.npy',
              # indexes_path = 'Data/Mouse/indexes.npz'
              ):
        # if stage in list(range(1,23)):
        #     self.test_set  = self.MouseDataset(tvt=stage, data_path=data_path, target_path=target_path)
        assert stage in ['fit', 'test'], 'stage must be `fit` or `test`'
        if stage == 'fit':
            self.train_set = self.MouseDataset(tvt='train', data_path=data_path, target_path=target_path, indexes_path=indexes_path)
            self.val_set   = self.MouseDataset(tvt='val', data_path=data_path, target_path=target_path, indexes_path=indexes_path)
        if stage == 'test':
            self.test_set  = self.MouseDataset(tvt='test', data_path=data_path, target_path=target_path, indexes_path=indexes_path)
    
    def train_dataloader(self):
            return DataLoader(self.train_set, self.batch_size, num_workers=2, shuffle=True)
    
    def val_dataloader(self):
            return DataLoader(self.val_set, self.batch_size, num_workers=2, shuffle=True)

    def test_dataloader(self):
            return DataLoader(self.test_set, self.batch_size, num_workers=2, shuffle=True)

    def _make_chromosome_based_split(
        filename="Data/Mouse/augmented/coords_v2.txt",
        filename_out="Data/Mouse/augmented/indexes.npz",
        ):
        assert os.path.exists(filename), 'coords file does not exist'
        
        coords = pickle.load(open(filename, "rb"))

        chromosomes_train = ['chr4', 'chr10', 'chr6', 'chr13', 'chr19', 'chr16', 'chr18', 'chr17', 'chr2', 'chr3', 'chr5', 'chr9', 'chr1', 'chr12', 'chr7', 'chr15']
        chromosomes_val = ['chr11', 'chr14']
        chromosomes_test = ['chrX', 'chr8']
        indexes_train, indexes_val, indexes_test = [], [], []
        for ch in chromosomes_train:
            indexes_train.append(np.where([c[0] == ch for c in coords])[0])
        for ch in chromosomes_val:
            indexes_val.append(np.where([c[0] == ch for c in coords])[0])
        for ch in chromosomes_test:
            indexes_test.append(np.where([c[0] == ch for c in coords])[0])
        indexes_train, indexes_val, indexes_test = \
            np.concatenate(indexes_train), np.concatenate(indexes_val), np.concatenate(indexes_test)
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

    def _upsample(
      ds_path = 'Data/Mouse/augmented/low_tensor_observed.npy',
      out_path = 'Data/Mouse/augmented/low_tensor_observed_upsampled.npy' 
       ):
      ds = np.load(ds_path)

      upsample = nn.Upsample(scale_factor=2, mode='bilinear')
      ds = upsample(torch.from_numpy(ds).unsqueeze(0)).squeeze(0)
      np.save(out_path, ds)
