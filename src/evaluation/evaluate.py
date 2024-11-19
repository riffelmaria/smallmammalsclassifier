import argparse
import os

from evaluation.evaluation_functions import (
    get_path_of_pickle,
    load_df_from_pickle,
    print_metrics,
)


def evaluate_classification(): ...


def evaluate_testdf():

    parser = argparse.ArgumentParser(
        prog="EvaluateTestDF",
        description="Evaluates model performance after training with a test dataset",
    )

    parser.add_argument(
        "model_path",
        help="Path to trained model, e.g. /path/to/model_name",
    )

    # parser.add_argument(
    #    "model_path_with_testdf",
    #    help="Path to test_df of trained model",
    # ) # das kann ich selbst im model ordner suchen!

    args = parser.parse_args()
    model_path = args.model_path
    model_name = model_path.split(os.path.sep)[-1]
    save_path_txt = model_path + "/eval_metrics_testdf.txt"
    # save_path_png = model_path + "/metric_plots.png"

    test_df = load_df_from_pickle(get_path_of_pickle(args.model_path_with_testdf))

    prediction_scores_df = load_df_from_pickle(
        f"{args.model_path_with_testdf}/prediction_from_testdf_{model_name}.pkl"
    )

    for classes in list(test_df.columns):
        print(classes, file=open(save_path_txt, "a"))
        print_metrics(
            test_df[classes],
            prediction_scores_df[classes].round(0),
            model_name,
            save_path_txt,
        )  # hier je die passenden spalten in die funktion übergeben und der datei  save path anhängen, mit col name


if __name__ == "__main__":
    evaluate_classification()

#### BIG TO DO
