import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import logging


@hydra.main(version_base='1.3', config_path='../../conf', config_name='config')
def load_data(cfg: DictConfig):
    transform = instantiate(cfg.dataset.transform)
    dataset = instantiate(cfg.dataset.load, transform=transform)
    dataloader = DataLoader(dataset, **cfg.dataset.data_loader)
    return dataloader


if __name__ == '__main__':
    try:
        load_data()
    except Exception as e:
        logging.exception(e)
        raise e
