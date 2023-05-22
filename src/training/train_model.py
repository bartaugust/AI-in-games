import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
import lightning.pytorch as pl

import logging


from src.data_preparation.load_data import load_data
from src.model_preparation.load_model import load_model
torch.set_float32_matmul_precision('high')


@hydra.main(version_base='1.3', config_path='../../conf', config_name='config')
def train(cfg: DictConfig):
    dataloader = load_data(cfg)
    model = load_model(cfg)
    model.fit(dataloader)
    # trainer = pl.Trainer()
    # trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        logging.exception(e)
        raise e
