import os
import wandb
from srcc.hyperparameter_optimization import hyperparameter_optimization
from dotenv import load_dotenv


load_dotenv()

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "3"
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# Specify the hyperparameters
parameter_dict = {
    "lr": {"values": [2e-5, 3e-5, 5e-5]},
    "epochs": {"values": [2,3,4,5]},
    "max_len": {"values": [64, 128, 256]},
    "model_name": {
        "values": [
            "lmsys/vicuna-13b-v1.5-16k",
            "bigscience/bloom-3b",
            "chavinlo/gpt4-x-alpaca",
            "CarperAI/stable-vicuna-13b-delta",
            "meta-llama/Llama-2-7b-hf",
            # "raruidol/ArgumentMining-EN-ARI-Debate",
            "databricks/dolly-v2-7b",
            "chkla/roberta-argument"



        ]
    },
}

sweep_config = {
    "method": "bayes",
    "metric": {"name": "mean_f1", "goal": "maximize", "target": 0.99},
    "parameters": parameter_dict,
    "early_terminate": {"type": "hyperband", "min_iter": 10},
}

wandb.login(key=os.environ["WANDB_API_KEY"], anonymous="allow")
# Run the hyperparameter optimization
sweep_id = 'i99nzmqc'
sweep_id = wandb.sweep(sweep_config, project="master-thesis")
# wandb.login(key=os.environ["WANDB_API_KEY"], anonymous="allow")

wandb.agent(sweep_id, function=hyperparameter_optimization, project="master-thesis")
