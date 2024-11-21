# Functions to evaluate trained models with a test dataset
import os
import numpy as np
import pandas as pd
from shared.files_manipulation_functions import dict_to_jsonfile


def load_df_from_pickle(pickle_path: str) -> pd.DataFrame:
    df = pd.read_pickle(pickle_path)
    return df


def get_path_of_pickle(pickle_path: str, pickle_name: str) -> str:
    for file in walk_pkl(pickle_path):
        if pickle_name in file:
            return file
        else: continue


def walk_pkl(folder_path: str):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".pkl")):
                yield os.path.join(root, file)



def calculate_confusion_matrix(ground_truth: pd.DataFrame, prediction: pd.DataFrame, label_columns: list):
    # Initialize a result matrix
    result_matrix = pd.DataFrame(
        index=label_columns, 
        columns=["TP", "FP", "FN", "TN", "Recall", "Precision", "Accuracy", "F1"]
    )

    # Iterate over each label column specified
    for label in label_columns:
        # Extract the columns for ground truth and prediction for this label
        actual = ground_truth[label]
        pred = prediction[label]
        
        # Calculate TP, FP, FN, TN
        TP = np.sum((actual == 1) & (pred == 1))
        FP = np.sum((actual == 0) & (pred == 1))
        FN = np.sum((actual == 1) & (pred == 0))
        TN = np.sum((actual == 0) & (pred == 0))

        # Calculate metrics
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
        
        # Store the result for this label
        result_matrix.loc[label] = [TP, FP, FN, TN, recall, precision, accuracy, f1]
    
    return result_matrix


def print_metrics(
    result_df, model_name, save_path, threshold: float
) -> None: 
    print(f"Evalution metrics from test dataset:\n", file=open(save_path, "a"))
    print(f"Model: {model_name}", file=open(save_path, "a"))
    print(f"Threshold: {threshold}", file=open(save_path, "a"))
    pd.display(result_df.to_string())
    #print(f"{variable_metric}:\n{variable_value}\n", file=open(save_path, "a"))

