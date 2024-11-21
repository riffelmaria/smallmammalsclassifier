import argparse
import os
from pathlib import Path

from evaluation.evaluation_functions import (
    get_path_of_pickle,
    load_df_from_pickle,
    calculate_confusion_matrix,
    print_metrics,
)


def evaluate_classification(): pass


def evaluate_testdf():

    parser = argparse.ArgumentParser(
        prog="EvaluateTestDF",
        description="Evaluates model performance after training with a test dataset",
    )

    parser.add_argument(
        "model_path",
        help="Path to trained model, e.g. /path/to/model_name",
    )

    parser.add_argument(
        "-t",
        "--testdf_path",
        help="Path to test_df.pkl and the prediction/classification of test_df from trained model",
    ) 

    args = parser.parse_args()
    model_path = args.model_path
    model_name = Path(model_path).stem

    if args.testdf_path is None:
        testdf_path = f"temp/{model_name}"
    else:
        testdf_path = args.testdf_path

    save_path = f"{testdf_path}/eval_metrics_testdf.txt" # JSON better?

    test_df = load_df_from_pickle(get_path_of_pickle(testdf_path, pickle_name="test_dataset"))

    classified_df = load_df_from_pickle(get_path_of_pickle(testdf_path, pickle_name="classified"))

    metrics_df = calculate_confusion_matrix(ground_truth=test_df, prediction= classified_df, label_columns=list(test_df.columns))
    print_metrics(result_df=metrics_df, model_name=model_name, save_path=save_path)
    

if __name__ == "__main__":
    evaluate_classification()

