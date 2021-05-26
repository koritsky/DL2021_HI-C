import os
import sys
import torch
from tqdm import tqdm
import numpy as np

from akita.models import ModelAkita, ModelWGraph
from akita.utils import get_dataloaders, from_upper_triu, convert_gr_to_rb

import matplotlib.pyplot as plt

if __name__ == "__main__":

    device = "cuda"

    # name = "akita"
    # name = "ours"
    name = "ours_symm"

    kwargs = {
        "target_crop": 0,
        "preds_triu": False,
    }
    if name == "akita":
        model = ModelAkita(**kwargs).to(device)
        model.load_state_dict(torch.load("./checkpoints/akita.pth", map_location=device))
    elif name == "ours":
        model = ModelWGraph(**kwargs).to(device)
        model.load_state_dict(torch.load("./checkpoints/ours.pth", map_location=device))
    elif name == "ours_symm":
        model = ModelWGraph(symmetrize=True, **kwargs).to(device)
        model.load_state_dict(torch.load("./checkpoints/ours_symm.pth", map_location=device))
    model.eval()

    
    _, _, dataloader = get_dataloaders(target_crop=0)

    predictions = []
    with torch.no_grad():
        for sequences, _, _ in tqdm(dataloader):
            predictions.append(model(sequences.to(device)).cpu().numpy())
    predictions = np.concatenate(predictions)