import os
import wandb
from src.hyperparameter_optimization import hyperparameter_optimization


if __name__ == "__main__":
    
    # Initialize Weights & Biases
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    # Specify the hyperparameters
    parameter_dict = {
        "lr": {"values": [1e-5, 2e-5, 3e-5, 5e-5]},
        "epochs": {"values": [2, 3, 4]},
        "batch_size": {"values": [8, 16, 32]},
        "max_len": {"values": [128, 256, 512]},
        "model_name": {"values": ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]}
    }

    sweep_config = {
        "method": "random",
        "metric": {
            "name": "val_f1",
            "goal": "maximize"
        },
        "parameters": parameter_dict,
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 10
        }
    }

    # Run the hyperparameter optimization
    sweep_id = wandb.sweep(sweep_config, project="master-thesis")
    wandb.agent(sweep_id, function=hyperparameter_optimization, count=5)
