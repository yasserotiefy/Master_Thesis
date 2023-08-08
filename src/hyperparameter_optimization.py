from sklearn.model_selection import StratifiedKFold
from .train_evaluate import train_model
from .data_preparation import create_data_loader
from transformers import AutoTokenizer
from types import SimpleNamespace
import wandb
from .argument_model import ArgumentModel
from .train_evaluate import evaluate_model
import os
import torch
from .data_preparation import load_and_process_data
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


file_path = "data/argument_relation_class.csv"  # replace with the path to your data file

# Load and process the data
df_train, df_val = load_and_process_data(file_path)



def hyperparameter_optimization(config=None, device=DEVICE):
    """Hyperparameter optimization using wandb sweeps.

    """

    with wandb.init(project="master-thesis", config=config): 
        best_f1 = 0
        best_accuracy = 0

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        config = wandb.config

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
            model.to(device)
            train_model(model, train_data_loader, val_data_loader, config.epochs, config.lr, wandb, device)
            accuracy, f1 = evaluate_model(model, val_data_loader, device)
            wandb.log({f"val_accuracy_{fold}": accuracy, f"val_f1_{fold}": f1})
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
        wandb.log({"mean_accuracy": mean_accuracy, "std_accuracy": std_accuracy})
        wandb.log({"mean_f1": mean_f1, "std_f1": std_f1})

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
