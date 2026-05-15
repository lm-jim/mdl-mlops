from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl

def train_model(model, data_module, model_name="main", batch_size=256, max_epochs=40, version="0", save_output=True):
    conv_model = model
    
    if save_output:
        logger = CSVLogger(
            save_dir=f"models/{model_name}/logs",
            version=version,
        )
    else:
        logger = None

    callbacks = []

    early_stopping_callback = EarlyStopping(
        monitor="train_loss",
        patience=5,
        mode="min"
    )
    callbacks.append(early_stopping_callback)

    if save_output:
        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss",
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
        logger=logger
    )

    trainer.fit(conv_model, datamodule=data_module)