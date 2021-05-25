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

neptune_logger = NeptuneLogger(
            #offline_mode=True,
            project_name='koritsky/DL2021-Bio',
            api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YTY4ZWY2ZC1jNzQxLTQ1ZTctYTM2My03YTZhNDQ5MTRlNzYifQ==',
            tags=['test_hypermodel_logging']
        )

#from akita import Akita
#from vehicle import Vehicle

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class HyperModel(pl.LightningModule):
    def __init__(self, akita_checkpoint=None, vehicle_checkpoint=None):
        super().__init__()

        self.akita = Dummy()
        if akita_checkpoint is not None:
            self.akita.load_state_dict(torch.load(akita_checkpoint))
        self.vehicle = Dummy()
        if vehicle_checkpoint is not None:
            self.vehicle.load_state_dict(torch.load(vehicle_checkpoint))

        self.head = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, padding=1)
        )

        #for awesome pictures
        self.mapper = cm.ScalarMappable(cmap=cm.RdBu_r)

    def configure_optimizers(self):
        all_params = list(self.akita.parameters()) + list(self.vehicle.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=1e-5)
        return [opt]

    def akita_forward(self, sequence):
        return self.akita(sequence)
    
    def vehicle_forward(self, low_img):
        return self.vehicle(low_img)

    def forward(self, sequence, low_img):

        akita_output = self.akita_forward(sequence)
        vehicle_output = self.vehicle_forward(low_img)
        
        combined_input = torch.cat([akita_output, vehicle_output], dim=1) #stack along the channel dimension
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

        akita_output = self.akita_forward(sequence)
        vehicle_output = self.vehicle_forward(low_img)
        
        combined_input = torch.cat([akita_output, vehicle_output], dim=1) #stack along the channel dimension
        output = self.head(combined_input)
        
        #akita_normalized_output = (akita_output.detach() + 2) / 4
        akita_loss = F.mse_loss(akita_output.detach(), high_img_akita)
        vehicle_loss = F.mse_loss(vehicle_output.detach(), high_img_vehicle)
        final_loss = F.mse_loss(output, high_img_vehicle) #will be passed for backward

        akita_metrics = self.calculate_metrics(akita_output, high_img_akita)
        vehicle_metrics = self.calculate_metrics(vehicle_output.detach(), high_img_vehicle)
        final_metrics = self.calculate_metrics(output.detach(), high_img_vehicle)
        
        return {"loss":final_loss,
                "akita_loss":akita_loss,
                "vehicle_loss":vehicle_loss,
                "final_metrics":final_metrics,
                "akita_metrics":akita_metrics,
                "vehicle_metrics":vehicle_metrics,
                "akita_output":akita_output.detach(),
                "vehicle_output":vehicle_output.detach(),
                "final_output":output.detach(),
                "high_img":high_img_vehicle,
                "high_img_akita":high_img_akita,
                }
        
    def _epoch_logging(self, outputs, phase):
        
        averaged_loss = np.mean([o['loss'].item() for o in outputs])
        averaged_akita_loss = np.mean([o['akita_loss'].item() for o in outputs])
        averaged_vehicle_loss = np.mean([o['vehicle_loss'].item() for o in outputs])

        akita_output = torch.cat([o['akita_output'] for o in outputs[:3]], dim=0).cpu()
        final_output = torch.cat([o['vehicle_output'] for o in outputs[:3]], dim=0).cpu()
        vehicle_output = torch.cat([o['final_output'] for o in outputs[:3]], dim=0).cpu()
        high_imgs = torch.cat([o['high_img'] for o in outputs[:3]], dim=0).cpu()
        high_imgs_akita = torch.cat([o['high_img_akita'] for o in outputs[:3]], dim=0).cpu()
        
        ###logging####
        self.logger.experiment.log_metric('{}/loss'.format(phase), averaged_loss) # x=self.current_epoch, y=averaged_loss)
        self.logger.experiment.log_metric('{}/akita_loss'.format(phase),averaged_akita_loss) # x=self.current_epoch,  y=averaged_akita_loss)
        self.logger.experiment.log_metric('{}/vehicle_loss'.format(phase), averaged_vehicle_loss) # x=self.current_epoch, y=averaged_vehicle_loss)

        ###save images###
        self.log_pictures(akita_img=akita_output,
                          vehicle_img=vehicle_output,
                          final_img=final_output,
                          high_img=high_imgs,
                          high_img_akita=high_imgs_akita,
                          phase=phase)
    
    def _construct_grid(self, img):

        bs = img.shape[0] #number of images
        fig, ax = plt.subplots(1, bs, sharey=True)
        for i in range(bs):
            ax[i].imshow(img[i])
            #ax[i].set_title("Place for metrics", fontsize=8)

            ax[i].text(0, 0, 'your legend', bbox={'facecolor': 'white', 'pad': 10})
        
        return fig

    def log_pictures(self, akita_img, vehicle_img, final_img, high_img, high_img_akita, phase):
        
        grid = self._construct_grid(self.get_colors(final_img)) #utils.make_grid(self.get_colors(final_img), nrow=2)
        self.logger.experiment.log_image('{}/final_img'.format(phase), self.current_epoch, grid)
        
        akita_img_normalized = (akita_img + 2) / 4
        grid = self._construct_grid(self.get_colors(akita_img_normalized)) #utils.make_grid(self.get_colors(akita_img), nrow=2)
        self.logger.experiment.log_image('{}/akita_img'.format(phase), self.current_epoch,  grid)
        
        grid = self._construct_grid(self.get_colors(vehicle_img)) #utils.make_grid(self.get_colors(vehicle_img), nrow=2)
        self.logger.experiment.log_image('{}/vehicle_img'.format(phase), self.current_epoch, grid)

        grid = self._construct_grid(self.get_colors(high_img)) #utils.make_grid(self.get_colors(high_img), nrow=2)
        self.logger.experiment.log_image('{}/high_img'.format(phase), self.current_epoch, grid)

        high_img_akita_normalized = (high_img_akita + 2) / 4
        grid = self._construct_grid(self.get_colors(high_img_akita_normalized)) #utils.make_grid(self.get_colors(high_img), nrow=2)
        self.logger.experiment.log_image('{}/high_img'.format(phase), self.current_epoch, grid)

        plt.clf()
    
    def calculate_metrics(self, y_pred, y_true):
        return [0, 0, 0, 0, 0]

    def get_colors(self, x):
        colorized_x = torch.Tensor(self.mapper.to_rgba(x))
        colorized_x = colorized_x.squeeze(1)
        #colorized_x = colorized_x.permute((0, -1, 1, 2))[:, :-1, :, :]
        return colorized_x



if __name__ == "__main__":
    
    model = HyperModel().cuda()

    input_1, input_2 = torch.randn((4, 1, 8, 8)).cuda(), torch.randn((4, 1, 8, 8)).cuda()

    output = model(input_1, input_2)
    print(output)
    print("output shape: ", output.shape)

    inps_1 = torch.randn((4, 1, 8, 8))
    inps_2 = torch.randn((4, 1, 8, 8))
    tgts = torch.randn((4, 1, 8, 8))
    dataset = torch.utils.data.TensorDataset(inps_1, inps_2, tgts, tgts)

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=12, pin_memory=True)
    trainer = pl.Trainer(logger=neptune_logger,
                        max_epochs=3,
                        gpus=1)
    trainer.fit(model, train_dataloader=loader, val_dataloaders=loader)
    
    
    
    # for l in loader:
    #     print(l[0].shape, l[1].shape)