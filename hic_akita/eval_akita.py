import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from akita.models import ModelAkita, ModelWGraph
from akita.utils import get_dataloaders, from_upper_triu


def get_scores(preds, targets):
    targets = np.reshape(targets, (-1,))
    preds = np.reshape(preds, (-1,))
    scores = {
        "mae": mean_absolute_error(targets, preds),
        "mse": mean_squared_error(targets, preds),
        "pearson": pearsonr(targets, preds)[0],
        "spearman": spearmanr(targets, preds)[0],
    }
    return scores


def run(model, dataloader):
    device = next(model.parameters()).device
    target_batches, target_tri_batches, pred_tri_batches = [], [], []
    with torch.no_grad():
        for sequences, targets, target_tri in tqdm(dataloader):
            predictions = model(sequences.to(device))
            pred_tri_batches.append(predictions.cpu().numpy())
            target_batches.append(targets.cpu().numpy())
            target_tri_batches.append(target_tri.cpu().numpy())
    targets = np.concatenate(target_batches)
    target_tri = np.concatenate(target_tri_batches)
    pred_tri = np.concatenate(pred_tri_batches)
    pred_imgs = []
    for p in pred_tri:
        pred_imgs.append(from_upper_triu(p, model.image_size, model.diagonal_offset))
    pred_imgs = np.stack(pred_imgs)
    return targets, target_tri, pred_imgs, pred_tri


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["akita", "ours", "ours_symm"])
    return parser.parse_args()

"""
akita
        mae   mse  pearson spearman
train 0.220 0.092    0.778    0.760
val   0.273 0.157    0.658    0.676
test  0.283 0.151    0.611    0.585

ours
      mae   mse   pearson  spearman
train 0.215 0.085    0.792    0.760
val   0.274 0.155    0.664    0.653
test  0.282 0.148    0.614    0.564

ours_symm
        mae   mse  pearson spearman
train 0.224 0.089    0.784    0.738
val   0.276 0.158    0.659    0.647
test  0.283 0.149    0.611    0.559
"""


if __name__ == "__main__":
    args = parse_args()
    dataloader_train, dataloader_val, dataloader_test = get_dataloaders(target_crop=10,)

    device = "cuda"
    if args.model == "akita":
        model = ModelAkita().to(device)
        filename_checkpoint = os.path.join("lightning_logs", "version_225", 
            "checkpoints", "epoch=88-step=85083.ckpt")
        filename_checkpoint_out = os.path.join("checkpoints", "akita.pth")
    elif args.model == "ours":
        model = ModelWGraph().to(device)
        filename_checkpoint = os.path.join("lightning_logs", "version_227",
            "checkpoints", "epoch=93-step=89863.ckpt")
        filename_checkpoint_out = os.path.join("checkpoints", "ours.pth")
    elif args.model == "ours_symm":
        model = ModelWGraph(symmetrize=True).to(device)
        filename_checkpoint = os.path.join("lightning_logs", "version_229",
            "checkpoints", "epoch=98-step=94643.ckpt")
        filename_checkpoint_out = os.path.join("checkpoints", "ours_symm.pth")

    model.load_state_dict({k[6:]: v for k, v in
        torch.load(filename_checkpoint, map_location=device)["state_dict"].items()})
    model.eval()
    # torch.save(model.state_dict(), filename_checkpoint_out)

    _, target_tri, _, pred_tri = run(model, dataloader_train)
    scores_train = get_scores(target_tri, pred_tri)
    _, target_tri, _, pred_tri = run(model, dataloader_val)
    scores_val = get_scores(target_tri, pred_tri)
    _, target_tri, _, pred_tri = run(model, dataloader_test)
    scores_test = get_scores(target_tri, pred_tri)
    print("{:5s} {:>5s} {:>5s} {:>8s} {:>8s}".format("", "mae", "mse", "pearson", "spearman"))
    print("{:5s} {:5.3f} {:5.3f} {:8.3f} {:8.3f}".format("train",
        scores_train["mae"], scores_train["mse"], scores_train["pearson"], scores_train["spearman"]))
    print("{:5s} {:5.3f} {:5.3f} {:8.3f} {:8.3f}".format("val",
        scores_val["mae"], scores_val["mse"], scores_val["pearson"], scores_val["spearman"]))
    print("{:5s} {:5.3f} {:5.3f} {:8.3f} {:8.3f}".format("test",
        scores_test["mae"], scores_test["mse"], scores_test["pearson"], scores_test["spearman"]))