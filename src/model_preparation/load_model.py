import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.model_preparation.vae import VAE
from torch.utils.data import DataLoader
import logging


@hydra.main(version_base='1.3', config_path='../../conf', config_name='config')
def load_model(cfg: DictConfig):
    model = instantiate(cfg.model.load,cfg)
    # model = instantiate(cfg.model.load)
    return model


if __name__ == '__main__':
    try:
        load_model()
    except Exception as e:
        logging.exception(e)
        raise e
