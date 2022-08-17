import argparse
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from akita.models import ModelAkita, ModelWGraph
from akita.utils import convert_gr_to_rb, from_upper_triu, get_dataloaders
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


class PLModel(pl.LightningModule):
    def __init__(self, model="akita", **kwargs):
        super().__init__()
        if model == "akita":
            self.model = ModelAkita(**kwargs)
        elif model == "ours":
            self.model = ModelWGraph(symmetrize=False, **kwargs)
        elif model == "ours_symm":
            self.model = ModelWGraph(symmetrize=True, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        sequences, targets, targets_tri = train_batch
        pred = self.model(sequences)
        loss = F.mse_loss(pred, targets_tri)
        self.log("train_loss", loss)
        return {
            "loss": loss,
            "pred": pred.detach().cpu(),
            "target": targets.cpu(),
            "target_tri": targets_tri.cpu(),
        }

    def backward(self, loss, optimizer, optimizer_idx):
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.)
        loss.backward()

    def validation_step(self, val_batch, batch_idx):
        sequences, targets, targets_tri = val_batch
        pred = self.model(sequences)
        loss = F.mse_loss(pred, targets_tri)
        self.log("val_loss", loss)
        return {
            "loss": loss,
            "pred": pred.detach().cpu(),
            "target": targets.cpu(),
            "target_tri": targets_tri.cpu(),
        }

    def log_images_epoch(self, outputs, mode="val"):
        pred_imgs, target_imgs = [], []
        for i in range(min(10, len(outputs))):
            pred = outputs[i]["pred"]
            target = outputs[i]["target"]
            for j in range(pred.shape[0]):
                img_pred = from_upper_triu(pred[j].cpu().numpy(
                ), self.model.image_size, self.model.diagonal_offset)
                img_pred = torch.from_numpy(img_pred)
                img_target = target[j]
                pred_imgs.append(convert_gr_to_rb(img_pred))
                target_imgs.append(convert_gr_to_rb(img_target))
        grid_pred = torchvision.utils.make_grid(pred_imgs, nrow=5)
        grid_target = torchvision.utils.make_grid(target_imgs, nrow=5)
        grid_both = torchvision.utils.make_grid(pred_imgs+target_imgs, nrow=10)
        self.logger.experiment.add_image(
            f"{mode}_pred", grid_pred, self.current_epoch)
        self.logger.experiment.add_image(
            f"{mode}_target", grid_target, self.current_epoch)
        self.logger.experiment.add_image(
            f"{mode}_imgs", grid_both, self.current_epoch)

    def validation_epoch_end(self, outputs):
        for k, v in self.get_val_scores(outputs).items():
            self.log(f"val_scores/{k}", v, self.current_epoch)
        return self.log_images_epoch(outputs, "val")

    def training_epoch_end(self, outputs):
        return self.log_images_epoch(outputs, "train")

    def get_val_scores(self, outputs):
        preds, targets = [], []
        for out in outputs:
            preds.append(out["pred"].numpy())
            targets.append(out["target_tri"].numpy())
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        targets = np.reshape(targets, (-1,))
        preds = np.reshape(preds, (-1,))
        scores = {
            "mae": mean_absolute_error(targets, preds),
            "mse": mean_squared_error(targets, preds),
            "pearson": pearsonr(targets, preds)[0],
            "spearman": spearmanr(targets, preds)[0],
        }
        return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["akita", "ours", "ours_symm"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataloader_train, dataloader_val, _ = get_dataloaders(target_crop=10,)
    model = PLModel(model=args.model)
    trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer.fit(model, dataloader_train, dataloader_val)
