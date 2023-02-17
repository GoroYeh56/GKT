from pathlib import Path

import logging
import pytorch_lightning as pl
import hydra

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback
from cross_view_transformer.tabular_logger import TabularLogger


log = logging.getLogger(__name__)

CONFIG_PATH = Path.cwd() / 'config'
CONFIG_NAME = 'config.yaml'


def maybe_resume_training(experiment):
    log.info(f'In maybe_resume_training:')
    log.info(f'experiment.ckptt: {experiment.ckptt}')
    save_dir = Path(experiment.save_dir).resolve()
    
    log.info(f'save_dir: {save_dir}')
    log.info(f'**/{experiment.uuid}/checkpoints/*.ckpt')
    checkpoints = list(save_dir.glob(
        f'**/{experiment.uuid}/checkpoints/*.ckpt'))
    log.info(f'checkpoints: {checkpoints}')
    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')

    return checkpoints[-1]

# config_path: currnet_working_directory/config ( => GKT/segmentation + /config)
# config_name: 'config.yaml'
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # Create and load model/data
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # Optionally load model
    ckpt_path = maybe_resume_training(cfg.experiment)

    if ckpt_path is not None:
        # ckpt_path: ~/GKT/segmentation/outputs/uuid_test/checkpoints/model_test.ckpt
        model_module.backbone = load_backbone(ckpt_path)

    # Loggers and callbacks
    logger = pl.loggers.TensorBoardLogger(save_dir=cfg.experiment.save_dir)
    tab_logger = TabularLogger(save_dir=cfg.experiment.save_dir)
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(filename='model',
                        every_n_train_steps=cfg.experiment.checkpoint_interval),

        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
    ]

    # Train
    trainer = pl.Trainer(logger=[logger, tab_logger],
                         callbacks=callbacks,
                         enable_progress_bar=False,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         **cfg.trainer)
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)
    trainer.validate(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
