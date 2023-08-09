import os
import wandb
from src.hyperparameter_optimization import hyperparameter_optimization

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['WANDB_AGENT_MAX_INITIAL_FAILURES'] = '3'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES']='0'

    
# Specify the hyperparameters
parameter_dict = {
    "lr": {"values": [2e-5]},
    "epochs": {"values": [2,3,4]},
    "batch_size": {"values": [128]},
    "max_len": {"values": [128]},
    "model_name": {"values": ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]},
}

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "mean_f1",
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
wandb.agent(sweep_id, function=hyperparameter_optimization)
