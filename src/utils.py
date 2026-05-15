import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pathlib import Path
import yaml
import logging

from src import model

def setup_logging(nivel : str):
    path_final = get_project_root()
    Path(path_final/"logs").mkdir(exist_ok=True)
    path_final = path_final / "logs"/ "mdl-mlops.log"

    logging.basicConfig(
        level = getattr(logging, nivel.upper(), logging.DEBUG),
        format = "%(asctime)s | %(levelname)-8s | %(funcName)s.%(lineno)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(path_final)
        ]
    )

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_config(nombre : str) -> dict:
    logger = logging.getLogger(__name__)
    logger.info("Cargando configuración...")
    raiz_proyecto = get_project_root()
    fichero_leer = raiz_proyecto / "config" / nombre

    with open(fichero_leer) as file:
        output = yaml.safe_load(file)
        logger.info("Carga de configuración completada")
        return output

def download_mnist():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

def load_best_model():
    config = load_config("configuration.yaml")
    return model.ConvCVAE.load_from_checkpoint(f"models/main/best_model-{config['model_version']}.ckpt")