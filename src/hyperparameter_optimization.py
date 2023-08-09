from sklearn.model_selection import StratifiedKFold
from .data_preparation import create_data_loader
from transformers import AutoTokenizer
import wandb
from .argument_model import ArgumentModel
import os
import torch
from .data_preparation import load_and_process_data
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


file_path = "data/argument_relation_class.csv"  # replace with the path to your data file

# Load and process the data
df_train, df_val = load_and_process_data(file_path)

# Get a list of device ids
pl.seed_everything(42)


def hyperparameter_optimization():
    """Hyperparameter optimization using wandb sweeps.

    """
    wandb.init(project="master-thesis")
    config = wandb.config
    current_id = wandb.run.id
    wandb.finish()
    wandb_logger = WandbLogger(project="master-thesis", id=current_id)

    best_f1 = 0
    best_accuracy = 0

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    val_accuracies = []
    val_f1_scores = []

    # implement cross validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df_train, y=df_train.label.values)):
        print(f"Fold: {fold}")
        train_data_loader = create_data_loader(df_train.iloc[train_idx], tokenizer, config.max_len, config.batch_size)
        val_data_loader = create_data_loader(df_train.iloc[val_idx], tokenizer, config.max_len, config.batch_size)

        model = ArgumentModel(config.model_name, config.lr)

        trainer = pl.Trainer(devices=1, accelerator='gpu', logger=wandb_logger, 
                                max_epochs=config.epochs, min_epochs=config.epochs, 
                                strategy="auto", log_every_n_steps=1)
        trainer.fit(model, train_data_loader, val_data_loader)
        metrics = trainer.validate(model, val_data_loader)
        accuracy = metrics[0]["val_acc_epoch"]
        f1 = metrics[0]["val_f1_epoch"]
        val_accuracies.append(accuracy)
        val_f1_scores.append(f1)

    val_accuracies = np.array(val_accuracies)
    val_f1_scores = np.array(val_f1_scores)

    # Calculate the mean and standard deviation
    mean_accuracy = val_accuracies.mean()
    std_accuracy = val_accuracies.std()
    mean_f1 = val_f1_scores.mean()
    std_f1 = val_f1_scores.std()

    # Log the mean and standard deviation to wandb
    wandb_logger.log_metrics({"mean_accuracy": mean_accuracy, "std_accuracy": std_accuracy})
    wandb_logger.log_metrics({"mean_f1": mean_f1, "std_f1": std_f1})

    if mean_f1 > best_f1:
        # Save the best model to wandb
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pt'))
        artifact_name = f"{wandb.run.id}_model"
        at = wandb.Artifact(artifact_name, type="model")
        at.add_file(os.path.join(wandb.run.dir, 'best_model.pt'))
        wandb.log_artifact(at, aliases=[f"best_model_{wandb.run.id}"])
        best_f1 = mean_f1
        best_accuracy = mean_accuracy
        

    return best_accuracy, best_f1
