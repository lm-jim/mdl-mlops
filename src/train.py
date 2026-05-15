from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
import wandb

def train_model(model, data_module, model_name="main", batch_size=256, max_epochs=40, version="0", save_output=True):
    conv_model = model
    
    loggers = []

    if save_output:
        csv_logger = CSVLogger(
            save_dir=f"models/{model_name}/logs",
            version=version,
        )
        loggers.append(csv_logger)

        wandb_logger = WandbLogger(
            project="mdl-mlops",
            name=f"{model_name}-v{version}",
            config={
                "model_name": model_name,
                "version": version,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "latent_dim": getattr(model, "latent_dim", "unknown"),
                "learning_rate": getattr(model, "lr", "unknown")
            }
        )
        loggers.append(wandb_logger)

    callbacks = []

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )
    callbacks.append(early_stopping_callback)

    if save_output:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=f"models/{model_name}",
            filename=f'best_model-{version}',
            enable_version_counter=False
        )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
        enable_checkpointing=True,
        logger=loggers
    )

    trainer.fit(conv_model, datamodule=data_module)

    if save_output:
        artifact = wandb.Artifact(
            name=f"{model_name}-v{version}", 
            type="model"
        )
        artifact.add_file(checkpoint_callback.best_model_path)
        wandb.log_artifact(artifact)