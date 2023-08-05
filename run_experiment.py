import os
import wandb
from src.data_preparation import load_and_process_data, create_data_loader
from src.hyperparameter_optimization import hyperparameter_optimization
import torch


if __name__ == "__main__":
    
    # Initialize Weights & Biases
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    file_path = "data/argument_relation_class.csv"  # replace with the path to your data file

    # Load and process the data
    df_train, df_val = load_and_process_data(file_path)

    # Specify the hyperparameters
    param_grid = {
        "model_name": ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],  # "bert-base-uncased", "roberta-base", "distilbert-base-uncased"
        "batch_size": [16],
        "max_len": [32,64,128],
        "epochs": [1],
        "lr": [3e-5]
    }


    # Run the hyperparameter optimization
    best_params_acc, best_param_f1, best_accuracy, best_f1 = hyperparameter_optimization(df_train, 
    df_val, param_grid, device=DEVICE)