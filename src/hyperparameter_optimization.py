from sklearn.model_selection import ParameterGrid
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


file_path = "data/argument_relation_class.csv"  # replace with the path to your data file

# Load and process the data
df_train, df_val = load_and_process_data(file_path)


def hyperparameter_optimization(config=None, device=DEVICE):
    
    best_f1 = 0
    best_accuracy = 0
    
    with wandb.init(project="master-thesis", config=config): 

        config = wandb.config

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        train_data_loader = create_data_loader(df_train, tokenizer, config.max_len, config.batch_size)
        val_data_loader = create_data_loader(df_val, tokenizer, config.max_len, config.batch_size)

        
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        model = ArgumentModel(config.model_name, config.lr)
        train_model(model, train_data_loader, val_data_loader, config.epochs, config.lr, wandb, device)
        accuracy, f1 = evaluate_model(model, val_data_loader, device)
        wandb.log({"val_accuracy": accuracy, "val_f1": f1})

        if f1 > best_f1:
            # Save the best model to wandb
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pt'))
            artifact_name = f"{wandb.run.id}_model"
            at = wandb.Artifact(artifact_name, type="model")
            at.add_file(os.path.join(wandb.run.dir, 'best_model.pt'))
            wandb.log_artifact(at, aliases=[f"best_model_{wandb.run.id}"])
        

    return best_accuracy, best_f1
