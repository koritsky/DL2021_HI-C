import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from torchvision import utils
from neptune.new.types import File

import itertools

neptune_logger = NeptuneLogger(
            #offline_mode=True,
            project_name='koritsky/DL2021-Bio',
            api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YTY4ZWY2ZC1jNzQxLTQ1ZTctYTM2My03YTZhNDQ5MTRlNzYifQ==',
            tags=['Graph head']
        )

import os, sys
sys.path.append(os.path.join(sys.path[0], "vehicle"))
sys.path.append(os.path.join(sys.path[0], "hic_akita"))
from vehicle.Models.VEHiCLE_Module import GAN_Model 

from hic_akita.akita.models import ModelWGraph
from hic_akita.akita.layers import Symmetrize2d

from dataloader import get_dataloaders
from metrics import get_scores

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class HyperModel(pl.LightningModule):
    def __init__(self,
     #akita_checkpoint=None,
     graph_checkpoint='hic_akita/checkpoints/ours_symm.pth',
      #vehicle_checkpoint=None,
      vehicle_checkpoint='vehicle/Weights/vehicle.ckpt'
      ):
        super().__init__()

        self.akita = ModelWGraph(target_crop=6, preds_triu=False, symmetrize=True) #Dummy()
        if graph_checkpoint is not None:
            self.akita.load_state_dict(torch.load(graph_checkpoint))
        self.vehicle = GAN_Model() #Dummy()

        if vehicle_checkpoint is not None:
            self.vehicle.load_state_dict(torch.load(vehicle_checkpoint)['state_dict'])
            pass

        self.head = nn.Sequential(
            nn.Conv2d(2, 16, 3, 1, padding=1),
            nn.ReLU(inplace=True),
           # nn.BatchNorm2d(16),
            Symmetrize2d(),

            nn.Conv2d(16, 1, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            Symmetrize2d()
        )

        #for awesome pictures
        self.mapper = cm.get_cmap('RdBu_r') #cm.ScalarMappable(cmap=cm.RdBu_r)

    def configure_optimizers(self):
        all_params = list(self.akita.parameters()) + list(self.head.parameters()) + list(self.vehicle.parameters())
        opt = torch.optim.AdamW(all_params, lr=1e-5, weight_decay=1e-5)
        return [opt]

    def akita_forward(self, sequence):
        return  self.akita(sequence).unsqueeze(1)
    
    def vehicle_forward(self, low_img):
        return self.vehicle(low_img)

    def forward(self, sequence, low_img):

        akita_output = self.akita_forward(sequence)
        vehicle_output = self.vehicle_forward(low_img.unsqueeze(1))
        
        combined_input = torch.cat([(akita_output + 2) / 4, vehicle_output], dim=1) #stack along the channel dimension
        output = self.head(combined_input)
        
        return output

    def training_step(self, batch, batch_idx):
        
        output = self._step(batch)

        return output

    def training_epoch_end(self, outputs):
        
        self._epoch_logging(outputs, phase='train')
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        
        output = self._step(batch)
        
        return output

    def validation_epoch_end(self, outputs):

        self._epoch_logging(outputs, phase='val')

    def _step(self, batch):

        sequence, low_img, high_img_akita, high_img_vehicle = batch
        #low_img = low_img.unsqueeze(1) #[bs, 1, 200, 200]
        high_img_akita = self.crop_img(high_img_akita.unsqueeze_(1), cropping=6)
        high_img_vehicle = self.crop_img(high_img_vehicle.unsqueeze_(1), cropping=6)

        akita_output = self.akita_forward(sequence) #[bs, 1, 188, 188]
        vehicle_output = self.vehicle_forward(low_img.unsqueeze_(1)) #[bs, 1, 188, 188]
        
        combined_input = torch.cat([(akita_output + 2) / 4, vehicle_output], dim=1) #stack along the channel dimension
        output = self.head(combined_input)
        
        #akita_normalized_output = (akita_output.detach() + 2) / 4
        akita_loss = F.mse_loss(akita_output.detach(), high_img_akita)
        vehicle_loss = F.mse_loss(vehicle_output.detach(), high_img_vehicle)
        final_loss = F.mse_loss(output, high_img_vehicle) #will be passed for backward

        akita_metrics = self.calculate_metrics(akita_output.detach(), high_img_akita.detach())
        vehicle_metrics = self.calculate_metrics(vehicle_output.detach(), high_img_vehicle.detach())
        final_metrics = self.calculate_metrics(output.detach(), high_img_vehicle.detach())
        
        return {"loss":final_loss,
                "akita_loss":akita_loss.detach(),
                "vehicle_loss":vehicle_loss.detach(),
                "final_metrics":final_metrics,
                "akita_metrics":akita_metrics,
                "vehicle_metrics":vehicle_metrics,
                "akita_output":akita_output.detach(),
                "vehicle_output":vehicle_output.detach(),
                "final_output":output.detach(),
                "high_img":high_img_vehicle.detach(),
                "high_img_akita":high_img_akita.detach(),
                }
        
    def _epoch_logging(self, outputs, phase):
        
        averaged_loss = np.mean([o['loss'].item() for o in outputs])
        averaged_akita_loss = np.mean([o['akita_loss'].item() for o in outputs])
        averaged_vehicle_loss = np.mean([o['vehicle_loss'].item() for o in outputs])

        #kludge to test metrics
        akita_metrics = []
        final_metrics = []
        vehicle_metrics = []
        for o in outputs:
            akita_metrics.append(o['akita_metrics'])
            final_metrics.append(o['final_metrics'])
            vehicle_metrics.append(o['vehicle_metrics'])

        
        
        #print("akita metrics: ", akita_metrics)

        akita_output = torch.cat([o['akita_output'] for o in outputs[:6]], dim=0).cpu()
        final_output = torch.cat([o['final_output'] for o in outputs[:6]], dim=0).cpu()
        vehicle_output = torch.cat([o['vehicle_output'] for o in outputs[:6]], dim=0).cpu()
        high_imgs = torch.cat([o['high_img'] for o in outputs[:6]], dim=0).cpu()
        high_imgs_akita = torch.cat([o['high_img_akita'] for o in outputs[:6]], dim=0).cpu()


        ###save images###
        self.log_pictures(akita_img=akita_output,
                          vehicle_img=vehicle_output,
                          final_img=final_output,
                          high_img=high_imgs,
                          high_img_akita=high_imgs_akita,
                          
                          akita_metrics=akita_metrics[:6],
                          vehicle_metrics=vehicle_metrics[:6],
                          final_metrics=final_metrics[:6],

                          phase=phase)
        ###logging####
        self.logger.experiment.log_metric('{}/loss'.format(phase), averaged_loss) # x=self.current_epoch, y=averaged_loss)
        self.logger.experiment.log_metric('{}/akita_loss'.format(phase),averaged_akita_loss) # x=self.current_epoch,  y=averaged_akita_loss)
        self.logger.experiment.log_metric('{}/vehicle_loss'.format(phase), averaged_vehicle_loss) # x=self.current_epoch, y=averaged_vehicle_loss)

        self.log_metrics(akita_metrics=akita_metrics,
                         final_metrics=final_metrics,
                         vehicle_metrics=vehicle_metrics,
                         
                         phase=phase)

    def log_metrics(self, akita_metrics, final_metrics, vehicle_metrics, phase):
        akita_metrics = list(itertools.chain(*akita_metrics))
        final_metrics = list(itertools.chain(*final_metrics))
        vehicle_metrics = list(itertools.chain(*vehicle_metrics))
        self.logger.experiment.log_metric('{}/final_mse'.format(phase), np.mean(final_metrics[::4]))
        self.logger.experiment.log_metric('{}/final_spearman'.format(phase), np.mean(final_metrics[1::4])) 
        self.logger.experiment.log_metric('{}/final_pearson'.format(phase), np.mean(final_metrics[2::4])) 
        self.logger.experiment.log_metric('{}/final_scc'.format(phase), np.mean(final_metrics[3::4]))

        self.logger.experiment.log_metric('{}/akita_mse'.format(phase), np.mean(akita_metrics[::4]))
        self.logger.experiment.log_metric('{}/akita_spearman'.format(phase), np.mean(akita_metrics[1::4])) 
        self.logger.experiment.log_metric('{}/akita_pearson'.format(phase), np.mean(akita_metrics[2::4])) 
        self.logger.experiment.log_metric('{}/akita_scc'.format(phase), np.mean(akita_metrics[3::4])) 

        self.logger.experiment.log_metric('{}/vehicle_mse'.format(phase), np.mean(vehicle_metrics[::4]))
        self.logger.experiment.log_metric('{}/vehicle_spearman'.format(phase), np.mean(vehicle_metrics[1::4])) 
        self.logger.experiment.log_metric('{}/vehicle_pearson'.format(phase), np.mean(vehicle_metrics[2::4])) 
        self.logger.experiment.log_metric('{}/vehicle_scc'.format(phase), np.mean(vehicle_metrics[3::4])) 

    
    def _construct_grid(self, img, metrics=None):

        bs = img.shape[0] #number of images
        fig, ax = plt.subplots(1, bs, sharey=True, figsize=(15*bs, 15))
        fig.tight_layout(pad=3.0)

        for i in range(bs):
            ax[i].imshow(img[i])
            #ax[i].set_title("Place for metrics", fontsize=8)
            
            ax[i].set_xticks([]) #off ticks
            ax[i].set_yticks([])

            if metrics is not None:

                metrics_str = "MSE: %.2f \nSpearman: %.2f \nPearson: %.2f \nSCC: %.2f" % tuple(metrics[i])
                ax[i].text(0, -0.25, metrics_str, transform=ax[i].transAxes, fontsize=55) #bbox={'facecolor': 'white', 'pad': 10})
        
        return fig

    def log_pictures(self, akita_img, vehicle_img, final_img, high_img, high_img_akita, phase, akita_metrics=None, vehicle_metrics=None, final_metrics=None):
        
        grid = self._construct_grid(self.get_colors(final_img), final_metrics) #utils.make_grid(self.get_colors(final_img), nrow=2)
        self.logger.experiment.log_image('{}/final_img'.format(phase), grid) #self.current_epoch, grid)
        
        akita_img_normalized = (akita_img + 2) / 4
        grid = self._construct_grid(self.get_colors(akita_img_normalized), akita_metrics) #utils.make_grid(self.get_colors(akita_img), nrow=2)
        self.logger.experiment.log_image('{}/akita_img'.format(phase), grid) #self.current_epoch,  grid)
        
        grid = self._construct_grid(self.get_colors(vehicle_img), vehicle_metrics) #utils.make_grid(self.get_colors(vehicle_img), nrow=2)
        self.logger.experiment.log_image('{}/vehicle_img'.format(phase), grid) #self.current_epoch, grid)

        grid = self._construct_grid(self.get_colors(high_img)) #utils.make_grid(self.get_colors(high_img), nrow=2)
        self.logger.experiment.log_image('{}/high_img'.format(phase), grid) #self.current_epoch, grid)

        high_img_akita_normalized = (high_img_akita + 2) / 4
        grid = self._construct_grid(self.get_colors(high_img_akita_normalized)) #utils.make_grid(self.get_colors(high_img), nrow=2)
        self.logger.experiment.log_image('{}/high_img_akita'.format(phase), grid) #self.current_epoch, grid)

        plt.clf()
    
    def calculate_metrics(self, y_pred, y_true):
        scores = get_scores(y_pred, y_true)
        return [scores['mse'], scores['spearman'], scores['pearson'], scores['scc']]

    def get_colors(self, x):
        colorized_x = torch.from_numpy(self.mapper(x.numpy()))
        colorized_x = colorized_x.squeeze(1)
        #colorized_x = colorized_x.permute((0, -1, 1, 2))[:, :-1, :, :]
        return colorized_x

    def from_upper_triu(self, flatten_triu, img_size=188, num_diags=2):
        
        print("flatten shape: ", flatten_triu.shape)
        z = torch.cat([torch.zeros((img_size, img_size)).unsqueeze(0) for _ in range(flatten_triu.shape[0])]) #[batch, img_size, img_size]
        triu_tup = torch.triu_indices(img_size, img_size, num_diags) #[2, number of elements in triu]
        z[:, triu_tup[0], triu_tup[1]] = flatten_triu
        
        return z + z.pertmute((0, 2, 1))

    def crop_img(self, img, cropping=0):
        _, _, s1, s2 = img.shape
        return img[:, :, 
            cropping:s1-cropping, 
            cropping:s2-cropping]

if __name__ == "__main__":
    
    model = HyperModel()

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(batch_size=1)

    trainer = pl.Trainer(logger=neptune_logger,
                        max_epochs=1,
                        gpus=1,
                        accumulate_grad_batches=32
                        )

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

    torch.save(model.state_dict(), "hypermodel-graph.pth")