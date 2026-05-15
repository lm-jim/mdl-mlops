from src import dataloaders, model, train, utils

import logging

def run_pipeline():
    
    config = utils.load_config("configuration.yaml")
    utils.setup_logging(config["log_level"])
    logger = logging.getLogger(__name__)
    
    logger.info("--- Iniciando pipeline de entrenamiento ---")

    logger.debug("Definiendo dataloaders y modelo...")
    dataloader = dataloaders.define_dataloaders(batch_size=config["data_batch_size"])
    conv_model = model.ConvCVAE(latent_dim=config["latent_dim"], lr=config["learning_rate"])
    logger.debug("Dataloaders y modelo definidos correctamente")
    
    logger.debug("Iniciando entrenamiento...")
    train.train_model(conv_model, dataloader, batch_size=config["train_batch_size"], max_epochs=config["epochs"], version=config["model_version"])
    logger.debug("Entrenamiento finalizado correctamente")

    logger.info("--- Pipeline de entrenamiento finalizado ---")

run_pipeline()