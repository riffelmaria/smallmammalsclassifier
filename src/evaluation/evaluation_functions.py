# Functions to evaluate my trained models
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def load_df_from_pickle(pickle_path):
    df = pd.read_pickle(pickle_path)
    return df


def get_path_of_pickle(model_path, pickle_name="test_df.pkl"):
    return os.path.join(model_path, pickle_name)


def walk_pkl(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".pkl")):
                yield os.path.join(root, file)


def calc_evaluation_metrics(test_df, prediction_df):

    evaluation_metrics = {
        "Confusion Matrix": confusion_matrix(test_df, prediction_df),
        "Accuracy": accuracy_score(test_df, prediction_df),
        "Precision": precision_score(test_df, prediction_df, average="macro"),
        "Recall": recall_score(test_df, prediction_df, average="macro"),
        "F1": f1_score(test_df, prediction_df, average="macro"),
    }
    return evaluation_metrics


def print_metrics(
    test_df, prediction_df, model_name, save_path
):  # diese funktion noch nutzen
    evaluation_metrics = calc_evaluation_metrics(test_df, prediction_df)
    for variable_metric in evaluation_metrics:
        variable_value = evaluation_metrics[variable_metric]
        print("Evaluation metrics from prediction\n", file=open(save_path, "a"))
        print(f"Model: {model_name}", file=open(save_path, "a"))
        print(f"{variable_metric}:\n{variable_value}\n", file=open(save_path, "a"))
