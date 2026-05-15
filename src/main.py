import wandb

from src import dataloaders, model, train, utils
import pytorch_lightning as pl
import logging

def run_pipeline():
    config = utils.load_config("gbl_config.yaml")
    model_config = utils.load_model_config()
    utils.setup_logging(config["log_level"])
    logger = logging.getLogger(__name__)
    


    pl.seed_everything(model_config["seed"], workers=True)

    logger.info("--- Iniciando pipeline de entrenamiento ---")

    logger.debug("iniciando Weights & Biases...")
    wandb.init(project="mdl-mlops", name=f"{model_config['model_name']}-v{model_config['model_version']}", config=model_config, job_type="training")
    logger.debug("Weights & Biases iniciado")

    logger.debug("Definiendo dataloaders y modelo...")
    dataloader = dataloaders.define_dataloaders(batch_size=model_config["data_batch_size"])
    conv_model = model.ConvCVAE(latent_dim=model_config["latent_dim"], lr=model_config["learning_rate"])
    logger.debug("Dataloaders y modelo definidos correctamente")
    
    logger.debug("Iniciando entrenamiento...")
    train.train_model(conv_model, dataloader, batch_size=model_config["train_batch_size"], max_epochs=model_config["epochs"], model_name=model_config["model_name"], version=model_config["model_version"])
    logger.debug("Entrenamiento finalizado correctamente")

    logger.info("--- Pipeline de entrenamiento finalizado ---")

run_pipeline()