from sklearn.model_selection import StratifiedKFold

from src.data_preparation import load_and_process_data
import numpy as np

import wandb
from datetime import timedelta
import wandb
from wandb import AlertLevel

import logging
from ludwig.api import LudwigModel
from dotenv import load_dotenv

import os

load_dotenv()


os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

file_path = (
    "data/argument_relation_class.csv"  # replace with the path to your data file
)

# Load and process the data
df_train = load_and_process_data(file_path)

# map labels to strings from 0,1 to Not_Related, Related
df_train["label"] = df_train["label"].map({0: "Not_Related", 1: "Related"})
print(df_train.head())

best_f1 = 0
best_accuracy = 0


def hyperparameter_optimization(config=None):
    """Hyperparameter optimization using wandb sweeps."""

    global best_f1
    global best_accuracy
    # Initialize the wandb logger
    with wandb.init(config=config):
        config = wandb.config
        current_id = wandb.run.id


        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        ludwig_config = {
                "input_features": [
                    {
                    "name": "text",    # The name of the input column
                    "type": "text",     # Data type of the input column
                    "encoder": {
                        "type": "auto_transformer",   # The model architecture to use
                        "pretrained_model_name_or_path": config.model_name,  # Pretrained model name or path
                        "trainable": False,
                         
                },
                                                 
                },
                ],
                "preprocessing": {
                        "max_sequence_length": config.max_len,  # Max length of the text inputs
                        "split": {
                            "type": "random",
                            "probabilities": [1.0, 0.0, 0.0],
                        }

                    },
                "output_features": [
                    {
                    "name": "label",
                    "type": "category",
                    }
                ],
                "trainer": {
                    "learning_rate": config.lr,  # learning rate of the optimizer
                    "epochs": config.epochs,  # We'll train for three epochs. Training longer might give
                    "batch_size": 64,  # Batch size of the training data
                }
            }

        val_accuracies = []
        val_f1_scores = []
        val_precision = []
        val_recall = []
        val_roc_auc = []
        val_loss = []
        truth = []
        preds = []
        label_probs = []

        model = LudwigModel(ludwig_config, logging_level=logging.INFO, gpus=[0])
        # implement cross validation
        for fold, (train_idx, val_idx) in enumerate(
            kf.split(X=df_train, y=df_train.label.values)
        ):
            
            train_data = df_train.iloc[train_idx]
            val_data = df_train.iloc[val_idx]

            
            train_stats, preprocessed_data, output_directory = model.train(dataset=train_data,
                                                                           skip_save_model=True,
                                                                           skip_save_progress=True,
                                                                           skip_save_log=True,
                                                                           skip_save_processed_input=True,
                                                                           output_directory="/home/")
            test_stats, predictions, output_directory = model.evaluate(
                                                                        val_data,
                                                                        collect_predictions=True,
                                                                        collect_overall_stats=True,
                                                                        skip_save_eval_stats=True,
                                                                        )

            # print(test_stats)
            # wandb.log(jsons.dump(train_stats))
            # wandb.log(jsons.dump(test_stats))
            accuracy = float(test_stats["label"]["accuracy"]) 
            f1 = float(test_stats["label"]["overall_stats"]["avg_f1_score_macro"])
            precision = float(test_stats["label"]["overall_stats"]["avg_precision_macro"])
            recall = float(test_stats["label"]["overall_stats"]["avg_recall_macro"])
            roc_auc = float(test_stats["label"]["roc_auc"])
            loss = float(test_stats["label"]["loss"])

            val_accuracies.append(accuracy)
            val_f1_scores.append(f1)
            val_precision.append(precision)
            val_recall.append(recall)
            val_roc_auc.append(roc_auc)
            val_loss.append(loss)
            

            wandb.log({f"fold_accuracy": accuracy, "fold_f1": f1}, step=fold)
            wandb.log(
                {f"fold_precision": precision, "fold_recall": recall}, step=fold
            )
            wandb.log({f"fold_roc_auc": roc_auc}, step=fold)
            wandb.log({f"fold_loss": loss}, step=fold)
            
            predictions, output_directory = model.predict(val_data["text"].to_frame(),
                                                          return_type="dict")

            preds.extend(predictions.label_predictions.values.astype(str))

            truth.extend(df_train.iloc[val_idx].label.values.astype(str))
            

            print("========================================= Done with fold =========================================")

        # Convert the lists to numpy arrays  
        val_accuracies = np.array(val_accuracies)
        val_f1_scores = np.array(val_f1_scores)
        val_precision = np.array(val_precision)
        val_recall = np.array(val_recall)
        val_roc_auc = np.array(val_roc_auc)
        val_loss = np.array(val_loss)



        # Calculate the mean and standard deviation
        mean_accuracy = val_accuracies.mean()
        std_accuracy = val_accuracies.std()
        mean_f1 = val_f1_scores.mean()
        std_f1 = val_f1_scores.std()
        mean_precision = val_precision.mean()
        std_precision = val_precision.std()
        mean_recall = val_recall.mean()
        std_recall = val_recall.std()
        mean_roc_auc = val_roc_auc.mean()
        std_roc_auc = val_roc_auc.std()
        mean_loss = val_loss.mean()
        std_loss = val_loss.std()


        # Log the mean and standard deviation to wandb
        wandb.log(
            {"mean_accuracy": mean_accuracy, "std_accuracy": std_accuracy}
        )
        wandb.log({"mean_f1": mean_f1, "std_f1": std_f1})
        wandb.log(
            {"mean_precision": mean_precision, "std_precision": std_precision}
        )
        wandb.log({"mean_recall": mean_recall, "std_recall": std_recall})
        wandb.log({"mean_roc_auc": mean_roc_auc, "std_roc_auc": std_roc_auc})
        wandb.log({"mean_loss": mean_loss, "std_loss": std_loss})


        print("========================================= Done with all folds =========================================")

        # Log the confusion matrix to wandb
 
        wandb.log({"Arg Relationship confusion matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=truth,
            preds=preds,
            class_names=["Related", "Not_Related"],
        )
        }
        )

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_accuracy = mean_accuracy
            print(f"New best F1 score======================================: {best_f1}")

            wandb.alert(
                title='Highest mean F1 score',
                text=f'Mean F1 score {mean_f1} is a highest F1 score so far.',
                level=AlertLevel.INFO,
                wait_duration=timedelta(minutes=5)
            )

        return best_accuracy, best_f1
