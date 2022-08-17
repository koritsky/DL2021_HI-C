import pdb
import sys

sys.path.append(".")
# from Data.GM12878_DataModule import GM12878Module
from Data.Mouse_DataModule import MouseModule
from Models.VEHiCLE_Module import GAN_Model
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger

# dm  = GM12878Module(batch_size=1, piece_size=269)
# dm.prepare_data()
batch_size = 1
dm = MouseModule(batch_size=batch_size)
dm.setup(stage='fit')

with_vae = True

model = GAN_Model(with_vae=with_vae, batch_size=batch_size)


neptune_logger = NeptuneLogger(
            project_name='koritsky/DL2021-Bio',
            api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YTY4ZWY2ZC1jNzQxLTQ1ZTctYTM2My03YTZhNDQ5MTRlNzYifQ==',
            params=model.params,
            experiment_name='with_vae' if with_vae else 'no_vae'
        )

trainer = Trainer(gpus=1, max_epochs=15, logger=neptune_logger)
trainer.fit(model, dm)

neptune_logger.experiment.stop()