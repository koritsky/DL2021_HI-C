import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import NeptuneLogger
from tqdm import tqdm

from dataloader import get_dataloaders
from metrics import get_scores

# neptune_logger = NeptuneLogger(
#             #offline_mode=True,
#             project_name='koritsky/DL2021-Bio',
#             api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YTY4ZWY2ZC1jNzQxLTQ1ZTctYTM2My03YTZhNDQ5MTRlNzYifQ==',
#             tags=['Two convolutions in head + Normalized akita output + no batchnorm + frozen vehicle']
#         )

_, _, test_set = get_dataloaders()


@torch.no_grad()
def main(model, test_set, gpu=None):

    trainer = pl.Trainer(  # logger=neptune_logger,
        max_epochs=30,
        # limit_test_batches=10,
        gpus=gpu,
    )

    trainer.test(model, test_dataloaders=test_set)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['conv', 'graph'])
    parser.add_argument(
        '--plot', help='number of model outputs to plot, if 0 does not plot at all', type=int, default=0)
    parser.add_argument(
        "--checkpoint", help="path to weights checkpoint file", default=None)
    parser.add_argument('--cuda', type=int, choices=[1, 0])
    args = parser.parse_args()

    if args.model == 'conv':
        from hypermodel import HyperModel
        model = HyperModel()
        if args.checkpoint is None:
            args.checkpoint = "hypermodel.pth"
        model.load_state_dict(torch.load(args.checkpoint))
    elif args.model == 'graph':
        from hypermodel_graph import HyperModel
        model = HyperModel()
        if args.checkpoint is None:
            args.checkpoint = "hypermodel-graph.pth"
        model.load_state_dict(torch.load(args.checkpoint))

    _, _, test_set = get_dataloaders()

    model.eval()
    if args.cuda:
        model.cuda()

    cuda = 1 if args.cuda else None
    main(model, test_set, gpu=cuda)

    if args.plot:
        outputs = []
        with torch.no_grad():
            for i, b in enumerate(test_set):
                seq, low_img = b[0], b[1]
                if args.cuda:
                    seq = seq.cuda()
                    low_img = low_img.cuda()
                output = model.forward(seq, low_img).detach().cpu()

                outputs.append(output)

                if i >= args.plot:
                    break

            outputs = torch.cat(outputs, dim=0)
            colorized_output = model.get_colors(outputs)
            figure = model._construct_grid(colorized_output)
            figure.savefig('model_output.png')
