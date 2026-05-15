from src import utils, preprocessing

import logging
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl

from torch.utils.data import DataLoader

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        self.train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, transform=self.transform
        )
        self.test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

def define_dataloaders(batch_size=64):
    logger = logging.getLogger(__name__)
    logger.debug("Cargando datasets de MNIST...")
    utils.download_mnist()
    logger.debug("Datasets de MNIST cargados correctamente")

    logger.debug("Preprocesando datos...")
    preprocessing.preprocess_data()
    logger.debug("Datos preprocesados correctamente")

    logger.debug("Creando DataModule...")
    data_module = MNISTDataModule(batch_size=batch_size)
    logger.debug("DataModule creado correctamente")
    return data_module