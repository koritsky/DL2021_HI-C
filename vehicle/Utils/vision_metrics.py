#the implementation of SSIM in this file is pulled from DeepHiC https://github.com/omegahh/DeepHiC
import argparse
import sys
from math import exp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.VEHiCLE_Module import GAN_Model
from scipy.stats import pearsonr, spearmanr

sys.path.append(".")
sys.path.append("../")
import glob
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm as tqdm
import yaml
from Data.GM12878_DataModule import GM12878Module
from Data.K562_DataModule import K562Module
from pytorch_lightning import Trainer
from sklearn.decomposition import PCA


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def _toimg(self, mat):
        m = torch.tensor(mat)
        # convert to float and add channel dimension
        return m.float().unsqueeze(0)

    def _tohic(self, mat):
        mat.squeeze_()
        return mat.numpy()#.astype(int)

    def gaussian(self, width, sigma):
        gauss = torch.Tensor([exp(-(x-width//2)**2 / float(2 * sigma**2)) for x in range(width)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel, sigma=3):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def gaussian_filter(self, img, width, sigma=3):
        img = _toimg(img).unsqueeze(0)
        _, channel, _, _ = img.size()
        window = self.create_window(width, channel, sigma)
        mu1 = F.conv2d(img, window, padding=width // 2, groups=channel)
        return _tohic(mu1)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


    def ssim(self, img1, img2, window_size=11, size_average=True):
        img1 = _toimg(img1).unsqueeze(0)
        img2 = _toimg(img2).unsqueeze(0)
        _, channel, _, _ = img1.size()
        window = self.create_window(window_size, channel)
        window = window.type_as(img1)

        return self._ssim(img1, img2, window, window_size, channel, size_average)



    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)



class VisionMetrics:
    def __init__(self):
        self.ssim         = SSIM()
        self.metric_logs = {
            "pre_pcc":[],
            "pas_pcc":[],
            "pre_spc":[],
            "pas_spc":[],
            "pre_psnr":[],
            "pas_psnr":[],
            "pre_ssim":[],
            "pas_ssim":[],
            "pre_mse":[],
            "pas_mse":[],
            "pre_snr":[],
            "pas_snr":[]
            }


    def _logSSIM(self, data, target, output):
        self.metric_logs['pre_ssim'].append(self.compareSSIM(data, target))
        self.metric_logs['pas_ssim'].append(self.compareSSIM(output, target))

    def _logPSNR(self, data, target, output):
        self.metric_logs['pre_psnr'].append(self.comparePSNR(data, target))
        self.metric_logs['pas_psnr'].append(self.comparePSNR(output, target))

    def _logPCC(self, data, target, output):
        self.metric_logs['pre_pcc'].append(self.comparePCC(data, target))
        self.metric_logs['pas_pcc'].append(self.comparePCC(output, target))

    def _logSPC(self, data, target, output):
        self.metric_logs['pre_spc'].append(self.compareSPC(data, target))
        self.metric_logs['pas_spc'].append(self.compareSPC(output, target))

    def _logMSE(self, data, target, output):
        self.metric_logs['pre_mse'].append(self.compareMSE(data, target))
        self.metric_logs['pas_mse'].append(self.compareMSE(output, target))

    def _logSNR(self, data, target, output):
        self.metric_logs['pre_snr'].append(self.compareSNR(data, target))
        self.metric_logs['pas_snr'].append(self.compareSNR(output, target))

    def compareSPC(self, a, b):
        return spearmanr(a[0][0], b[0][0], axis=None)[0]

    def comparePCC(self, a, b):
        return pearsonr(a[0][0].flatten(), b[0][0].flatten())[0]

    def comparePSNR(self, a, b):
        MSE = np.square(a[0][0]-b[0][0]).mean().item()
        MAX = torch.max(b).item()
        return 20*np.log10(MAX) - 10*np.log10(MSE)

    def compareSNR(self, a, b):
        return torch.sum(b[0][0]).item()/torch.sqrt(torch.sum((b[0][0]-a[0][0])**2)).item()

    def compareSSIM(self, a, b):
        return self.ssim(a, b).item()

    def compareMSE(self, a, b):
        return np.square(a[0][0]-b[0][0]).mean().item()

    def log_means(self, name):
        return (name, np.mean(self.metric_logs[name]))

    def setDataset(self, chro, res=10000, piece_size=269, cell_line="GM12878"):
        if cell_line == "GM12878":
            self.dm_test      = GM12878Module(batch_size=1, res=res, piece_size=piece_size)
        if cell_line == "K562":
            self.dm_test      = K562Module(batch_size=1, res=res, piece_size=piece_size)
        self.dm_test.prepare_data()
        self.dm_test.setup(stage=chro)

    def getMetrics(self, model, spliter):
        self.metric_logs = {
            "pre_pcc":[],
            "pas_pcc":[],
            "pre_spc":[],
            "pas_spc":[],
            "pre_psnr":[],
            "pas_psnr":[],
            "pre_ssim":[],
            "pas_ssim":[],
            "pre_mse":[],
            "pas_mse":[],
            "pre_snr":[],
            "pas_snr":[]
            }

        for e, epoch in enumerate(tqdm(self.dm_test.test_dataloader())):
            print(str(e)+"/"+str(self.dm_test.test_dataloader().dataset.data.shape[0]))
            data, full_target, info = epoch
            target                  = full_target[:,:,6:-6,6:-6]
            filter_data             = data[:,:,6:-6,6:-6]
            if spliter == "vehicle" or spliter == "large":  #no need to seperate pieces
                output                  = model(data).detach()

            if spliter == "hicplus" or spliter == "hicsr":  #separater into 40x40 windows
                output   = torch.zeros((1,1,269,269))
                for i in range(0, 269-40, 28):
                    for j in range(0,269-40,28):
                        temp = data[:,:,i:i+40, j:j+40]
                        output[:,:,i+6:i+34, j+6:j+34] = model(temp)
                output = output[:,:,6:-6,6:-6].detach()
            
            if spliter == "deephic" or spliter=='vae':
                output   = torch.zeros((1,1,269,269))
                for i in range(0, 269-40, 28):
                    for j in range(0,269-40,28):
                        temp = data[:,:,i:i+40, j:j+40]
                        output[:,:,i+6:i+34, j+6:j+34] = model(temp)[:,:,6:-6,6:-6]
                output = output[:,:,6:-6,6:-6].detach()

            if spliter == "large_deephic":
                output  = model(data).detach()[:,:,6:-6,6:-6]


            self._logPCC(data=filter_data, target=target, output=output)
            self._logSPC(data=filter_data, target=target, output=output)
            self._logMSE(data=filter_data, target=target, output=output)
            self._logPSNR(data=filter_data, target=target, output=output)
            self._logSNR(data=filter_data, target=target, output=output)
            self._logSSIM(data=filter_data, target=target, output=output)
        print(list(map(self.log_means, self.metric_logs.keys())))
        return self.metric_logs

if __name__=='__main__':
    visionMetrics = VisionMetrics()
    visionMetrics.setDataset(20, cell_line="K562")
    WEIGHT_PATH   = "deepchromap_weights.ckpt"
    model         = GAN_Model()
    pretrained_model = model.load_from_checkpoint(WEIGHT_PATH)
    pretrained_model.freeze()
    visionMetrics.getMetrics(model=pretrained_model, spliter=False)
