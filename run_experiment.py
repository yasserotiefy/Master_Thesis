import os
import wandb
from src.hyperparameter_optimization import hyperparameter_optimization
from dotenv import load_dotenv
import torch


load_dotenv()

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "3"
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_DISABLE_CODE"] = "True"


torch.set_float32_matmul_precision('high')

# Specify the hyperparameters
parameter_dict = {
    "lr": {"values": [2e-5, 3e-5, 5e-5]},
    "epochs": {"values": [2, 3, 4, 5]},
    "max_len": {"values": [64, 128, 256]},
    "model_name": {
        "values": [
            # "bert-base-uncased",
            # "roberta-base",
            # "distilbert-base-uncased",
            # "raruidol/ArgumentMining-EN-ARI-Debate",
            # "raruidol/ArgumentMining-EN-AC-Essay-Fin",
            # # "raruidol/ArgumentMining-EN-AC-Financial",
            # # "raruidol/ArgumentMining-EN-CN-ARI-Essay-Fin",
            # "chkla/roberta-argument",
            # "lmsys/vicuna-13b-v1.5-16k",
            # "bigscience/bloom-7b",
            # "bigscience/bloom-560m",
            # "bigscience/bloomz-1b1",
            # "bigscience/bloomz-7b1",
            # "bigscience/bloom-7b1",
            # "meta-llama/Llama-2-7b-hf",
            # "reciprocate/vicuna-13b_rm_oasst-hh",
            # "lmsys/vicuna-13b-v1.5",
            # "DebateLabKIT/argument-analyst",
            # "addy88/argument-classifier",
            # "chavinlo/gpt4-x-alpaca",
            # "nickmuchi/finbert-tone-finetuned-finance-topic-classification",
            # "nickmuchi/deberta-v3-base-finetuned-finance-text-classification",
            # "ProsusAI/finbert",
            # "ahmedrachid/FinancialBERT-Sentiment-Analysis",
            # "NLPScholars/Roberta-Earning-Call-Transcript-Classification",
            # "chavinlo/gpt4-x-alpaca",
            # "CarperAI/stable-vicuna-13b-delta",
            # "meta-llama/Llama-2-7b-hf",
            # "databricks/dolly-v2-7b",
        ]
    },
}

sweep_config = {
    "method": "bayes",
    "metric": {"name": "mean_f1", "goal": "maximize", "target": 0.99},
    "parameters": parameter_dict,
    "early_terminate": {"type": "hyperband", "min_iter": 10},
}

# Run the hyperparameter optimization
sweep_id = 'lt59j2xu'
sweep_id = wandb.sweep(sweep_config, project="master-thesis")
wandb.login(key=os.environ["WANDB_API_KEY"], anonymous="allow")

wandb.agent(sweep_id, function=hyperparameter_optimization, project="master-thesis")
