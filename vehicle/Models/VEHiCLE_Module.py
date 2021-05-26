import torch.nn.functional as F
import torch.nn as nn
from . import VehicleGAN as  vgan 
import torch
import pdb
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from Utils.loss import vae_loss as vl
from Utils.loss import insulation as ins
import matplotlib.pyplot as plt
# from neptune.new.types import File

class GAN_Model(pl.LightningModule):
    def __init__(self, with_vae=True,batch_size=1):
        super(GAN_Model, self).__init__()
        self.mse_lambda     = 1
        self.tad_lambda     = 1
        self.vae_lambda     = 1e-3 
        self.gan_lambda     = 2.5e-3
        self.G_lr           = 5e-5
        self.D_lr           = 1e-6
        self.beta_1         = 0.9  
        self.beta_2         = 0.99
        self.num_res_blocks = 15
        self.generator      = vgan.Generator(num_res_blocks=self.num_res_blocks)
        self.discriminator  = vgan.Discriminator() 
        self.generator.init_params()
        self.discriminator.init_params()
        self.bce            = nn.BCEWithLogitsLoss()
        self.mse  = nn.L1Loss()
        self.vae_yaml        = "Weights/vehicle_vae_hparams.yaml"
        self.vae_weight      = "Weights/vehicle_vae_sk.ckpt"
        self.with_vae = with_vae
        if self.with_vae:
            self.vae             = vl.VaeLoss(self.vae_yaml, self.vae_weight)
        self.tad             = ins.InsulationLoss()
        self.is_sanity_check = True

        self.params = {
            'mse_lambda': self.mse_lambda,
            'tad_lambda': self.tad_lambda,
            'vae_lambda': self.vae_lambda,
            'gan_lambda': self.gan_lambda,
            'G_lr': self.G_lr,
            'D_lr': self.D_lr,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'num_res_blocks': self.num_res_blocks,
            'with_vae': self.with_vae,
            'vae_yaml': self.vae_yaml,
            'vae_weight': self.vae_weight,
            'batch_size': batch_size
        }

        self.val_losses = []
    
    def forward(self, x):
        fake = self.generator(x)
        return fake

    def tad_loss(self, target, output):

        return self.tad(target, output)

    def vae_loss(self, target, output):
        return self.vae(target, output)

    def adversarial_loss(self, target, output):
        return self.bce(target, output)

    def meanSquaredError_loss(self, target, output):
        return self.mse(target, output)

    def training_step(self, batch, batch_idx, optimizer_idx):
        data, full_target = batch
        target = full_target[:,:,6:-6,6:-6]

        #Generator
        if optimizer_idx == 0:
            self.generator.zero_grad()
            output      = self.generator(data)
            MSE_loss    = self.meanSquaredError_loss(output, target)
            
            TAD_loss    = self.tad_loss(output, target)
            pred_fake   = self.discriminator(output)
            labels_real = torch.ones_like(pred_fake, requires_grad=False)
            GAN_loss    = self.adversarial_loss(pred_fake, labels_real)

            if self.with_vae:
                VAE_loss    = self.vae_loss(output, target)
                total_loss_G = (self.tad_lambda * TAD_loss) + (self.vae_lambda * VAE_loss) + (self.mse_lambda * MSE_loss) + (self.gan_lambda * GAN_loss)
            else:
                total_loss_G = (self.tad_lambda * TAD_loss) + (self.mse_lambda * MSE_loss) + (self.gan_lambda * GAN_loss)
            self.logger.experiment.log_metric('train/total_loss_G', total_loss_G)
            self.logger.experiment.log_metric('train/TAD_loss', TAD_loss)
            self.logger.experiment.log_metric('train/MSE_loss', MSE_loss)
            self.logger.experiment.log_metric('train/GAN_loss', GAN_loss)
            self.logger.experiment.log_metric('train/VAE_loss', VAE_loss)
            return total_loss_G
        
        #Discriminator
        if optimizer_idx == 1:
            self.discriminator.zero_grad()
            #train on real data
            pred_real       = self.discriminator(target)
            labels_real     = torch.ones_like(pred_real, requires_grad=False)
            pred_labels_real = (pred_real>0.5).float().detach()
            acc_real        = (pred_labels_real == labels_real).float().sum() / labels_real.shape[0]
            loss_real       = self.adversarial_loss(pred_real, labels_real)
            # self.logger.experiment.log_metric('train/loss_real', loss_real)

            
            #train on fake data
            output           = self.generator(data)
            pred_fake        = self.discriminator(output.detach())
            labels_fake      = torch.zeros_like(pred_fake, requires_grad=False)
            pred_labels_fake = (pred_fake > 0.5).float()
            acc_fake         = (pred_labels_fake == labels_fake).float().sum()/labels_fake.shape[0]
            loss_fake        = self.adversarial_loss(pred_fake, labels_fake)
            # self.logger.experiment.log_metric('train/loss_fake', loss_fake)


            total_loss_D = loss_real + loss_fake
            self.logger.experiment.log_metric('train/total_loss_D', total_loss_D)
            return total_loss_D
            
        

    def validation_step(self, batch, batch_idx):
        data, full_target = batch
        output       = self.generator(data)
        target       = full_target[:,:,6:-6,6:-6]
        MSE_loss    = self.meanSquaredError_loss(output, target)

        self.val_losses.append(MSE_loss)

        if batch_idx == 5:
          self.logger.experiment.log_image('val/target_img', self.get_img(data, target, output, MSE_loss))
          # self.logger.experiment.log_image('val/enhanced_img', output[0, 0].cpu().numpy()) 

        return MSE_loss
      
    def on_validation_epoch_end(self):
      if self.is_sanity_check:
        self.exp_dir = self.trainer.logger.name + '/' + self.trainer.logger.version + '/'
        os.makedirs(self.exp_dir, exist_ok=True)
        self.examples_dir = self.exp_dir + 'examples/'
        os.makedirs(self.examples_dir, exist_ok=True)
        self.is_sanity_check = False
      else:

        avg_MSE_loss = sum(self.val_losses) / len(self.val_losses)
        self.logger.experiment.log_metric('val/avg_MSE_loss', avg_MSE_loss)
        
        checkpoint_path = f'{self.exp_dir}checkpoints/epoch-{self.current_epoch}_mse-{avg_MSE_loss}.ckpt'
        self.trainer.save_checkpoint(checkpoint_path)
        # self.logger.experiment.log_artifact('checkpoints', checkpoint_path)
      self.val_losses = []

      

       
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.G_lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.D_lr)
        return [opt_g, opt_d]

    def get_img(self, low, target, enhanced, MSE_loss):
        fig, ax = plt.subplots(1,3, figsize=(30, 10))
        fig.suptitle('MSE = ' + str(MSE_loss.cpu().numpy()))

        ax[0].imshow(low[0,0].cpu().numpy(),  cmap="Reds")
        ax[1].imshow(enhanced[0,0].cpu().numpy(),  cmap="Reds")
        ax[2].imshow(target[0,0].cpu().numpy(),  cmap="Reds")

        ax[0].set_title('low')
        ax[1].set_title('enhanced')
        ax[2].set_title('target')

        path = self.examples_dir + str(self.current_epoch) + '.png'
        plt.savefig(path)
        return path

