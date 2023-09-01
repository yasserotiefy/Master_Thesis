#!/usr/bin/env python

# # Simple Model Training Example
#
# This is a simple example of how to use the LLM model type to train
# a zero shot classification model. It uses the facebook/opt-350m model
# as the base LLM model.

# Import required libraries
import logging
import shutil

import os

from dotenv import load_dotenv

import pandas as pd
import yaml

from ludwig.api import LudwigModel

from src.data_preparation import load_and_process_data

import wandb


load_dotenv()
# clean out prior results
shutil.rmtree("./results", ignore_errors=True)

file_path = (
    "data/argument_relation_class.csv"  # replace with the path to your data file
)

# Load and process the data
df = load_and_process_data(file_path)

# map labels to strings from 0,1 to Not_Related, Related
df["label"] = df["label"].map({0: "Not_Related", 1: "Related"})

# sample 100 rows
df = df.sample(10)
print(df.tail(10)) 



wandb.login(key=os.environ["WANDB_API_KEY"], anonymous="allow")

with wandb.init(project="master-thesis", job_type="zero_shot_batch_prediction") as run:
    # Define the model configuration
    config = yaml.safe_load(
        """
    model_type: llm
    base_model: bigscience/bloom-3b
    generation:
        temperature: 0.1
        top_p: 0.75
        top_k: 40
        num_beams: 4
        max_new_tokens: 5
    prompt:
        task: "The task is about relationship classification between claim and premise in the domain of argumentation mining. 
        Classify the sample input as either Related or Not_Related, please return the label without any additional text."
    input_features:
    -
        name: text
        type: text
    output_features:
    -
        name: label
        type: category
        decoder:
            type: category_extractor
            match:
                "Not_Related":
                    type: contains
                    value: "Not_Related"
                "Related":
                    type: contains
                    value: "Related"
        """
    )

    # Define Ludwig model object that drive model training
    model = LudwigModel(config=config, logging_level=logging.INFO)

    # initiate model training
    (
        train_stats,  # dictionary containing training statistics
        preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
        output_directory,  # location of training results stored on disk
    ) = model.train(
        dataset=df, experiment_name="simple_experiment", model_name="simple_model", skip_save_processed_input=True
    )

    training_set, val_set, test_set, _ = preprocessed_data

    # batch prediction
    preds, _ = model.predict(test_set, skip_save_predictions=False)
    print(preds)

    test_set = test_set.to_df()
    print(test_set)

    wandb.log({"batch_prediction": wandb.Table(dataframe=preds)})

    # log confusion matrix, precision, recall, f1 score, roc curve, and PR curve
    wandb.sklearn.plot_confusion_matrix(
        truth=test_set["label"].values,
        preds=preds["label_predictions"].values,
        class_names=["Not_Related", "Related"],
    )
    wandb.sklearn.plot_precision_recall(
        truth=test_set["label"].values,
        probs=preds["label_predictions"].values,
        class_names=["Not_Related", "Related"],
    )
    wandb.sklearn.plot_roc(
        truth=test_set["label"].values,
        probs=preds["label_predictions"].values,
        class_names=["Not_Related", "Related"],
    )


