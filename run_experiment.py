import os
import wandb
from src.hyperparameter_optimization import hyperparameter_optimization
from dotenv import load_dotenv


load_dotenv()

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['WANDB_AGENT_MAX_INITIAL_FAILURES'] = '3'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES']='1'

    
# Specify the hyperparameters
parameter_dict = {
    "lr": {"values": [2e-5, 3e-5, 5e-5]},
    "epochs": {"values": [2, 3, 4, 5]},
    "batch_size": {"values": [32, 64, 128]},
    "max_len": {"values": [64, 128, 256]},
    "model_name": {"values": ["bert-base-uncased", "roberta-base",
                               "distilbert-base-uncased"]}
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
# sweep_id = wandb.sweep(sweep_config, project="master-thesis")
# wandb.login(key=os.environ["WANDB_API_KEY"], anonymous="allow")
sweep_id = '7qqvyt02'
wandb.agent(sweep_id, function=hyperparameter_optimization, project="master-thesis")
