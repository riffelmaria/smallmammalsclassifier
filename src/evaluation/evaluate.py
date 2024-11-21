import argparse
import os
from pathlib import Path

from evaluation.evaluation_functions import (
    get_path_of_pickle,
    load_df_from_pickle,
    calculate_confusion_matrix,
    print_metrics,
)

from classification.classification_functions import use_decision_threshold


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

    parser.add_argument(
        "--no-display",
        action="store_false",
        dest="display",
        help="flag, if no additional results from other thresholds should be displayed",
    )

    args = parser.parse_args()
    model_path = args.model_path
    model_name = Path(model_path).stem

    if args.testdf_path is None:
        testdf_path = f"temp/{model_name}"
    else:
        testdf_path = args.testdf_path

    save_path = f"{testdf_path}/eval_metrics_testdf.txt" # TODO JSON better?

    test_df = load_df_from_pickle(get_path_of_pickle(testdf_path, pickle_name="test_dataset"))

    classified_df = load_df_from_pickle(get_path_of_pickle(testdf_path, pickle_name="classified")) # TODO column "threshold" needs to be treated
    classified_threshold = classified_df["threshold"][0]

    print("These are evaluation results from using the standard or custum threshold:\n")

    metrics_df = calculate_confusion_matrix(ground_truth=test_df, prediction= classified_df.remove("threshold"), label_columns=list(test_df.columns))
    print_metrics(result_df=metrics_df, model_name=model_name, save_path=save_path, threshold = classified_threshold)

    if args.display:
        q = input("Do you want to display the evaluation results from other thresholds? (type 'yes')")
        if q == "yes":
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            predicted_df = load_df_from_pickle(get_path_of_pickle(testdf_path, pickle_name="prediction"))
            print("These are results from different thresholds:\n")
            for threshold in thresholds:
                new_df = use_decision_threshold(predicted_df, threshold=threshold)
                metrics_df = calculate_confusion_matrix(ground_truth=test_df, prediction= new_df.remove("threshold"), label_columns=list(test_df.columns))
                print_metrics(result_df=metrics_df, model_name=model_name, save_path=save_path, threshold=threshold)    

if __name__ == "__main__":
    evaluate_classification()

