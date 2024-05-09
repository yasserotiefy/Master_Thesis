from sklearn.model_selection import StratifiedKFold
from .data_preparation import create_data_loader
from transformers import AutoTokenizer
from .argument_model import ArgumentModel
import os
import torch
from .data_preparation import load_and_process_data
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import timedelta
import wandb
from wandb import AlertLevel

import sklearn
# os.environ["TOKENIZERS_PARALLELISM"] = "true"


file_path = (
    "data/orientation-gb-train.tsv"  # replace with the path to your data file
)

# Load and process the data
df_train = load_and_process_data(file_path)
df_test = load_and_process_data("data/orientation-gb-test.tsv")

# Get a list of device ids
pl.seed_everything(42)

best_f1 = 0
best_accuracy = 0
batch_size = 128

preds_parliament = []



def hyperparameter_optimization(config=None):
    """Hyperparameter optimization using wandb sweeps."""

    global best_f1
    global best_accuracy
    global preds_parliament
    # Initialize the wandb logger
    with wandb.init(config=config):
        config = wandb.config
        current_id = wandb.run.id

    wandb_logger = WandbLogger(project="master-thesis", id=current_id, log_model=False, save_dir="/home/master-thesis/")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name,
                                              token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    
    # LLama3 model has a different padding token
    tokenizer.pad_token = tokenizer.eos_token

    val_accuracies = []
    val_f1_scores = []
    val_precision = []
    val_recall = []
    truth = []
    preds = []
    

    model = ArgumentModel(config.model_name, config.lr)

    # implement cross validation
    for fold, (train_idx, val_idx) in enumerate(
        kf.split(X=df_train, y=df_train.label.values)
    ):
        
        train_data_loader = create_data_loader(
            df_train.iloc[train_idx], tokenizer, config.max_len, batch_size
        )
        val_data_loader = create_data_loader(
            df_train.iloc[val_idx], tokenizer, config.max_len, batch_size
        )
        


        trainer = pl.Trainer(
            accelerator="auto",
            logger=wandb_logger,
            max_epochs=config.epochs,
            min_epochs=config.epochs,
            strategy="auto",
            log_every_n_steps=1,
            enable_checkpointing=False,
        )
        trainer.fit(model, train_data_loader, val_data_loader)
        metrics = trainer.validate(model, val_data_loader)

        accuracy = metrics[0]["val_acc"]
        f1 = metrics[0]["val_f1"]
        precision = metrics[0]["val_precision"]
        recall = metrics[0]["val_recall"]
        val_accuracies.append(accuracy)
        val_f1_scores.append(f1)
        val_precision.append(precision)
        val_recall.append(recall)

        wandb_logger.log_metrics({f"fold_accuracy": accuracy, "fold_f1": f1}, step=fold)
        wandb_logger.log_metrics(
            {f"fold_precision": precision, "fold_recall": recall}, step=fold
        )

        pred = trainer.predict(model, val_data_loader)
        for tensor in pred:
            preds.extend(tensor.tolist())

        truth.extend(df_train.iloc[val_idx].label.values)
        
        test_data_loader = create_data_loader(
        df_test, tokenizer, config.max_len, batch_size
        )
        
    
        pred_parli = trainer.predict(model, test_data_loader)
        
        pred_parliament = []
        for tensor in pred_parli:
            pred_parliament.extend(tensor.tolist())        
        
        
        

    val_accuracies = np.array(val_accuracies)
    val_f1_scores = np.array(val_f1_scores)
    val_precision = np.array(val_precision)
    val_recall = np.array(val_recall)

    # Calculate the mean and standard deviation
    mean_accuracy = val_accuracies.mean()
    std_accuracy = val_accuracies.std()
    mean_f1 = val_f1_scores.mean()
    std_f1 = val_f1_scores.std()
    mean_precision = val_precision.mean()
    std_precision = val_precision.std()
    mean_recall = val_recall.mean()
    std_recall = val_recall.std()

    
    # truth_parliament = df_test.label.values
    # parliament_test_f1_score = sklearn.metrics.f1_score(truth_parliament, pred_parliament)
    
    # wandb_logger.log_metrics({"parliament_test_f1_score": parliament_test_f1_score})
    
    df_test["pred_parliament"] = pred_parliament
    df_test.to_csv("parliemant_predictions.csv", index=False)



    # Log the mean and standard deviation to wandb
    wandb_logger.log_metrics(
        {"mean_accuracy": mean_accuracy, "std_accuracy": std_accuracy}
    )
    wandb_logger.log_metrics({"mean_f1": mean_f1, "std_f1": std_f1})
    wandb_logger.log_metrics(
        {"mean_precision": mean_precision, "std_precision": std_precision}
    )
    wandb_logger.log_metrics({"mean_recall": mean_recall, "std_recall": std_recall})
    
    


    # Log the confusion matrix to wandb
    wandb_logger.experiment.log(
        {
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=truth,
                preds=preds,
                class_names=["Not_Related", "Related"],
            )
        }
    )

    # wandb_logger.experiment.log(
    #     {
    #         "PR_Curve": wandb.plot.pr_curve(
    #             probs=None,
    #             y_true=truth,
    #             preds=preds,
    #             class_names=["Not_Related", "Related"],
    #         )
    #     }
    # )

    # wandb_logger.experiment.log(
    #     {
    #         "ROC_Curve": wandb.plot.roc_curve(
    #             y_true=truth,
    #             preds=preds,
    #             labels=["Not_Related", "Related"],
    #         )
    #     }
    # )

    if mean_f1 > best_f1:
        # Save the best model to wandb
        # torch.save(
        #     model.state_dict(),
        #     os.path.join(wandb_logger.experiment.dir, "best_model.pt"),
        # )
        # artifact_name = f"{wandb_logger.experiment.id}_model"
        # at = wandb.Artifact(artifact_name, type="model")
        # at.add_file(os.path.join(wandb_logger.experiment.dir, "best_model.pt"))
        # wandb_logger.experiment.log_artifact(
        #     at, aliases=[f"best_model_{wandb_logger.experiment.id}"]
        # )
        best_f1 = mean_f1
        best_accuracy = mean_accuracy

        wandb.alert(
            title='High F1 score',
            text=f'F1 score {mean_f1} is a highest F1 score so far.',
            level=AlertLevel.INFO,
            wait_duration=timedelta(minutes=5)
        )

    torch.cuda.empty_cache()

    return best_accuracy, best_f1
