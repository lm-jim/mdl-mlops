import pytest
import os
from src import dataloaders, utils, model, train

# python.exe -m pytest tests/test_integration.py

def test_pipeline_execution():
    config = utils.load_config("configuration.yaml")

    dataloader = dataloaders.define_dataloaders(batch_size=config["data_batch_size"])
    conv_model = model.ConvCVAE(latent_dim=config["latent_dim"], lr=config["learning_rate"])
    
    train.train_model(conv_model, dataloader, batch_size=config["train_batch_size"], max_epochs=1, save_output=False)