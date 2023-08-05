from sklearn.model_selection import ParameterGrid
from .train_evaluate import train_model
from .data_preparation import create_data_loader
from transformers import AutoTokenizer
from types import SimpleNamespace
import wandb
from .argument_model import ArgumentModel
from .train_evaluate import evaluate_model
import os


def hyperparameter_optimization(df_train, df_val, param_grid, device="cpu"):
    param_grid = ParameterGrid(param_grid)
    best_params = None
    best_f1 = 0
    best_accuracy = 0
    


    for params in param_grid:
        print(f"Training with parameters: {params}")
        wandb_params = SimpleNamespace(
            **params
        )

        tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
        train_data_loader = create_data_loader(df_train, tokenizer, params['max_len'], params['batch_size'])
        val_data_loader = create_data_loader(df_val, tokenizer, params['max_len'], params['batch_size'])

        wandb.init(project="master-thesis", config=wandb_params)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        model = ArgumentModel(params['model_name'], params['lr'])
        train_model(model, train_data_loader, val_data_loader, params['epochs'], params['lr'], wandb, device)
        accuracy, f1 = evaluate_model(model, val_data_loader, wandb, device)

        if f1 > best_f1:
            best_f1 = f1
            best_params_f1 = params
            wandb.log({"best_f1": best_f1})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params_acc = params
            wandb.log({"best_accuracy": best_accuracy})

        wandb.finish()
    return best_params_acc, best_params_f1, best_accuracy, best_f1
