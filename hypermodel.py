import sys
import os
sys.path.append(os.path.join(sys.path[0], "vehicle"))
sys.path.append(os.path.join(sys.path[0], "hic_akita"))

from vehicle.Models.VEHiCLE_Module import GAN_Model
from metrics import get_scores
from hic_akita.akita.models import ModelAkita
from hic_akita.akita.layers import Symmetrize2d
from dataloader import get_dataloaders

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from matplotlib import cm
from pytorch_lightning.loggers import NeptuneLogger
from torchvision import utils

logger_params = yaml.safe_load(open("logger_params.yaml", "r"))
neptune_logger = NeptuneLogger(
    project=logger_params['project'],
    api_token=logger_params['api_token'],
    tags=['Another training run'],
)


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class HyperModel(pl.LightningModule):
    def __init__(self,
                 # akita_checkpoint=None,
                 akita_checkpoint='hic_akita/checkpoints/akita.pth',
                 # vehicle_checkpoint=None,
                 vehicle_checkpoint='vehicle/Weights/vehicle.ckpt'
                 ):
        super().__init__()

        self.akita = ModelAkita(target_crop=6, preds_triu=False)  # Dummy()
        map_location = None if torch.cuda.is_available() else "cpu"
        if akita_checkpoint is not None:
            self.akita.load_state_dict(
                torch.load(akita_checkpoint, map_location=map_location)
            )
        self.vehicle = GAN_Model()  # Dummy()

        if vehicle_checkpoint is not None:
            self.vehicle.load_state_dict(
                torch.load(vehicle_checkpoint, map_location=map_location)['state_dict']
            )

        self.head = nn.Sequential(
            nn.Conv2d(2, 16, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(16),
            Symmetrize2d(),

            nn.Conv2d(16, 1, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            Symmetrize2d()
        )

        # for awesome pictures
        # cm.ScalarMappable(cmap=cm.RdBu_r)
        self.mapper = cm.get_cmap('RdBu_r')


    def configure_optimizers(self):
        all_params = list(self.akita.parameters(
        )) + list(self.head.parameters()) + list(self.vehicle.parameters())
        opt = torch.optim.AdamW(all_params, lr=1e-5, weight_decay=1e-5)
        return [opt]


    def akita_forward(self, sequence):
        return self.akita(sequence).unsqueeze(1)


    # @torch.no_grad()
    def vehicle_forward(self, low_img):
        return self.vehicle(low_img)


    def forward(self, sequence, low_img):
        akita_output = self.akita_forward(sequence)
        vehicle_output = self.vehicle_forward(low_img.unsqueeze(1))
        # stack along the channel dimension
        combined_input = torch.cat(
            [(akita_output + 2 / 4), vehicle_output], dim=1)
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
        # low_img = low_img.unsqueeze(1) #[bs, 1, 200, 200]
        high_img_akita = self.crop_img(
            high_img_akita.unsqueeze_(1), cropping=6)
        high_img_vehicle = self.crop_img(
            high_img_vehicle.unsqueeze_(1), cropping=6)

        akita_output = self.akita_forward(sequence)  # [bs, 1, 188, 188]
        vehicle_output = self.vehicle_forward(
            low_img.unsqueeze_(1))  # [bs, 1, 188, 188]

        # stack along the channel dimension
        combined_input = torch.cat(
            [(akita_output + 2) / 4, vehicle_output], dim=1)
        output = self.head(combined_input)

        #akita_normalized_output = (akita_output.detach() + 2) / 4
        akita_loss = F.mse_loss(akita_output.detach(), high_img_akita)
        vehicle_loss = F.mse_loss(vehicle_output.detach(), high_img_vehicle)
        # will be passed for backward
        final_loss = F.mse_loss(output, high_img_vehicle)

        akita_metrics = self.calculate_metrics(
            akita_output.detach(), high_img_akita.detach())
        vehicle_metrics = self.calculate_metrics(
            vehicle_output.detach(), high_img_vehicle.detach())
        final_metrics = self.calculate_metrics(
            output.detach(), high_img_vehicle.detach())

        return {"loss": final_loss,
                "akita_loss": akita_loss.detach(),
                "vehicle_loss": vehicle_loss.detach(),
                "final_metrics": final_metrics,
                "akita_metrics": akita_metrics,
                "vehicle_metrics": vehicle_metrics,
                "akita_output": akita_output.detach(),
                "vehicle_output": vehicle_output.detach(),
                "final_output": output.detach(),
                "high_img": high_img_vehicle.detach(),
                "high_img_akita": high_img_akita.detach(),
                }

    def _epoch_logging(self, outputs, phase):

        averaged_loss = np.mean([o['loss'].item() for o in outputs])
        averaged_akita_loss = np.mean(
            [o['akita_loss'].item() for o in outputs])
        averaged_vehicle_loss = np.mean(
            [o['vehicle_loss'].item() for o in outputs])

        # kludge to test metrics
        akita_metrics = []
        final_metrics = []
        vehicle_metrics = []
        for o in outputs:
            akita_metrics.append(o['akita_metrics'])
            final_metrics.append(o['final_metrics'])
            vehicle_metrics.append(o['vehicle_metrics'])

        #print("akita metrics: ", akita_metrics)

        akita_output = torch.cat([o['akita_output']
                                 for o in outputs[:6]], dim=0).cpu()
        final_output = torch.cat([o['final_output']
                                 for o in outputs[:6]], dim=0).cpu()
        vehicle_output = torch.cat([o['vehicle_output']
                                   for o in outputs[:6]], dim=0).cpu()
        high_imgs = torch.cat([o['high_img']
                              for o in outputs[:6]], dim=0).cpu()
        high_imgs_akita = torch.cat([o['high_img_akita']
                                    for o in outputs[:6]], dim=0).cpu()

        ###save images###
        self.log_pictures(
            akita_img=akita_output,
            vehicle_img=vehicle_output,
            final_img=final_output,
            high_img=high_imgs,
            high_img_akita=high_imgs_akita,
            akita_metrics=akita_metrics[:6],
            vehicle_metrics=vehicle_metrics[:6],
            final_metrics=final_metrics[:6],
            phase=phase
        )
        ###logging####
        self.log(f"{phase}/loss", averaged_loss)
        self.log(f"{phase}/akita_loss", averaged_akita_loss)
        self.log(f"{phase}/vehicle_loss", averaged_vehicle_loss)

        self.log_metrics(
            akita_metrics=akita_metrics,
            final_metrics=final_metrics,
            vehicle_metrics=vehicle_metrics,
            phase=phase
        )


    def log_metrics(self, akita_metrics, final_metrics, vehicle_metrics, phase):
        akita_metrics = list(itertools.chain(*akita_metrics))
        final_metrics = list(itertools.chain(*final_metrics))
        vehicle_metrics = list(itertools.chain(*vehicle_metrics))
        for name, metrics in zip(
                ["final", "akita", "vehicle"],
                [final_metrics, akita_metrics, vehicle_metrics]
                ):
            self.log(f'{phase}/{name}_mse', np.mean(metrics[::4]))
            self.log(f'{phase}/final_spearman'.format(phase), np.mean(metrics[1::4]))
            self.log(f'{phase}/final_pearson'.format(phase), np.mean(metrics[2::4]))
            self.log(f'{phase}/final_scc'.format(phase), np.mean(metrics[3::4]))


    def _construct_grid(self, img, metrics=None):
        bs = img.shape[0]  # number of images
        fig, ax = plt.subplots(1, bs, sharey=True, figsize=(15*bs, 15))
        fig.tight_layout(pad=3.0)

        for i in range(bs):
            ax[i].imshow(img[i])
            #ax[i].set_title("Place for metrics", fontsize=8)

            ax[i].set_xticks([])  # off ticks
            ax[i].set_yticks([])

            if False and metrics is not None:
                metrics_str = "MSE: %.2f \nSpearman: %.2f \nPearson: %.2f \nSCC: %.2f" % tuple(
                    metrics[i])
                # bbox={'facecolor': 'white', 'pad': 10})
                ax[i].text(0, -0.25, metrics_str, transform=ax[i].transAxes, fontsize=55)
        return fig


    def log_pictures(self, akita_img, vehicle_img, final_img, high_img, high_img_akita, phase, akita_metrics=None, vehicle_metrics=None, final_metrics=None):
        # utils.make_grid(self.get_colors(final_img), nrow=2)
        grid = self._construct_grid(self.get_colors(final_img), final_metrics)
        # self.current_epoch, grid)
        self.logger.experiment['{}/final_img'.format(phase)].log(grid)

        akita_img_normalized = (akita_img + 2) / 4
        # utils.make_grid(self.get_colors(akita_img), nrow=2)
        grid = self._construct_grid(self.get_colors(
            akita_img_normalized), akita_metrics)
        # self.current_epoch,  grid)
        self.logger.experiment['{}/akita_img'.format(phase)].log(grid)

        # utils.make_grid(self.get_colors(vehicle_img), nrow=2)
        grid = self._construct_grid(
            self.get_colors(vehicle_img), vehicle_metrics)
        # self.current_epoch, grid)
        self.logger.experiment['{}/vehicle_img'.format(phase)].log(grid)

        # utils.make_grid(self.get_colors(high_img), nrow=2)
        grid = self._construct_grid(self.get_colors(high_img))
        # self.current_epoch, grid)
        self.logger.experiment['{}/high_img'.format(phase)].log(grid)

        high_img_akita_normalized = (high_img_akita + 2) / 4
        # utils.make_grid(self.get_colors(high_img), nrow=2)
        grid = self._construct_grid(self.get_colors(high_img_akita_normalized))
        # self.current_epoch, grid)
        self.logger.experiment['{}/high_img_akita'.format(phase)].log(grid)
        plt.clf()


    def calculate_metrics(self, y_pred, y_true):
        scores = get_scores(y_pred, y_true)
        return [scores['mse'], scores['spearman'], scores['pearson'], scores['scc'], ]


    def get_colors(self, x):
        colorized_x = torch.from_numpy(self.mapper(x.numpy()))
        colorized_x = colorized_x.squeeze(1)
        #colorized_x = colorized_x.permute((0, -1, 1, 2))[:, :-1, :, :]
        return colorized_x


    def from_upper_triu(self, flatten_triu, img_size=188, num_diags=2):
        print("flatten shape: ", flatten_triu.shape)
        z = torch.cat([torch.zeros((img_size, img_size)).unsqueeze(
            0) for _ in range(flatten_triu.shape[0])])  # [batch, img_size, img_size]
        # [2, number of elements in triu]
        triu_tup = torch.triu_indices(img_size, img_size, num_diags)
        z[:, triu_tup[0], triu_tup[1]] = flatten_triu

        return z + z.pertmute((0, 2, 1))

    def crop_img(self, img, cropping=0):
        _, _, s1, s2 = img.shape
        return img[:, :,
                   cropping:s1-cropping,
                   cropping:s2-cropping]

    @torch.no_grad()
    def test_step(self, batch, batch_idx):

        _, _, _, high_img_vehicle = batch
        high_img_vehicle = self.crop_img(
            high_img_vehicle.unsqueeze(1), cropping=6)

        output = self._step(batch)['final_output']

        # manually change diagonal, because it should be 1 always
        indices = [i for i in range(output.shape[3])]  # [0...188]
        output[:, :, indices, indices] = high_img_vehicle[:, :, indices, indices]

        output[:, :, indices[1:], indices[:1]
               ] = high_img_vehicle[:, :, indices[1:], indices[:1]]
        output[:, :, indices[:1], indices[1:]
               ] = high_img_vehicle[:, :, indices[:1], indices[1:]]

        # metrics
        scores = self.calculate_metrics(output, high_img_vehicle)
        return scores


    def test_epoch_end(self, outputs):
        print("mean mse: ", np.mean([score[0] for score in outputs]))
        print("mean spearman: ", np.mean([score[1] for score in outputs]))
        print("mean pearson: ", np.mean([score[2] for score in outputs]))
        print("mean scc: ", np.mean([score[3] for score in outputs]))
        #print("mean mae: ", np.mean([score[4] for score in outputs]))
        return outputs


if __name__ == "__main__":

    model = HyperModel()

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        batch_size=1)
    
    trainer = pl.Trainer(
        logger=neptune_logger,
        max_epochs=30,
        gpus=1,
        accumulate_grad_batches=32,
        
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), "hypermodel_frozen_vehicle.pth")
