from logging import config

import pytest
import os
from src import dataloaders, utils, model, train, preprocessing

# python.exe -m pytest tests/test_unit.py

def test_mnsit_download():
    utils.download_mnist()

def test_load_config():
    config = utils.load_config("configuration.yaml")
    assert "seed" in config
    assert "epochs" in config
    assert "data_batch_size" in config
    assert "train_batch_size" in config
    assert "latent_dim" in config
    assert "learning_rate" in config

def test_config_formats():
    config = utils.load_config("configuration.yaml")
    assert isinstance(config["seed"], int)
    assert isinstance(config["epochs"], int)
    assert isinstance(config["data_batch_size"], int)
    assert isinstance(config["train_batch_size"], int)
    assert isinstance(config["latent_dim"], int)
    assert isinstance(config["learning_rate"], float)

def test_mnist_data():
    assert os.path.exists('./data/MNIST/raw/train-images-idx3-ubyte.gz')
    assert os.path.exists('./data/MNIST/raw/train-labels-idx1-ubyte.gz')
    assert os.path.exists('./data/MNIST/raw/t10k-images-idx3-ubyte.gz')
    assert os.path.exists('./data/MNIST/raw/t10k-labels-idx1-ubyte.gz')

def test_preprocess_data():
    preprocessing.preprocess_data()

def test_dataloader_creation():
    config = utils.load_config("configuration.yaml")
    data_module = dataloaders.define_dataloaders(batch_size=config["data_batch_size"])
    assert data_module is not None

def test_model_creation():
    config = utils.load_config("configuration.yaml")
    conv_model = model.ConvCVAE(latent_dim=config["latent_dim"], lr=config["learning_rate"])
    assert conv_model is not None

def test_train_model():
    config = utils.load_config("configuration.yaml")
    data_module = dataloaders.define_dataloaders(batch_size=config["data_batch_size"])
    conv_model = model.ConvCVAE(latent_dim=config["latent_dim"], lr=config["learning_rate"])
    train.train_model(
                        conv_model, 
                        data_module, 
                        model_name='unit_test_model', 
                        batch_size=config["train_batch_size"], 
                        max_epochs=1, 
                        save_output=False
                    )
    
def test_best_model_exists():
    config = utils.load_config("configuration.yaml")
    assert os.path.exists(f"models/main/best_model-{config['model_version']}.ckpt")

def test_best_model_loadable():
    conv_model = utils.load_best_model()
    conv_model.eval()
    assert conv_model is not None