import pdb
import sys

sys.path.append(".")
from Data.GM12878_DataModule import GM12878Module
from Data.Mouse_DataModule import MouseModule
from Models.VAE_Module import VAE_Model
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

# dm      = GM12878Module()
# dm.prepare_data()

dm = MouseModule()
dm.setup(stage='fit')
# dm.setup(stage=14)

pargs = {'batch_size': 512,
        'condensed_latent': 3,
        'gamma': 1.0, 
        'kld_weight': .000001,
        'kld_weight_inc': 0.000,
        'latent_dim': 200,
        'lr': 0.00001,
        'pre_latent': 4608}

neptune_logger = NeptuneLogger(
    project_name='koritsky/DL2021-Bio',
    api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YTY4ZWY2ZC1jNzQxLTQ1ZTctYTM2My03YTZhNDQ5MTRlNzYifQ==',
    params=pargs,
    experiment_name='only_vae'
)

checkpoint_callback = ModelCheckpoint(
          filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
          save_top_k=-1
          )
model    = VAE_Model(batch_size=pargs['batch_size'],
                    condensed_latent=pargs['condensed_latent'],
                    gamma=pargs['gamma'],
                    kld_weight=pargs['kld_weight'],
                    kld_weight_inc=pargs['kld_weight_inc'],
                    latent_dim=pargs['latent_dim'],
                    lr=pargs['lr'],
                    pre_latent=pargs['pre_latent'])
                    
trainer = Trainer(gpus=1, max_epochs=200, logger=neptune_logger, callbacks=[checkpoint_callback])
trainer.fit(model, dm)

'''
pargs = {'batch_size': 512,
        'condensed_latent': 3,
        'gamma': 1.0, 
        'kld_weight': .0001,
        'kld_weight_inc': 0.000,
        'latent_dim': 110,
        'lr': 0.00001,
        'pre_latent': 4608}
'''
