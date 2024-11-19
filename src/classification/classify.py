# predict:
# samples – the files to generate predictions for. Can be: - a dataframe with index containing audio paths, OR - a dataframe with multi-index (file, start_time, end_time), OR - a list (or np.ndarray) of audio file paths

import argparse
import json
import os
from pathlib import Path

from classification.classification_functions import (
    create_audio_df,
    prediction,
    use_decision_threshold,
)
from shared.audio_functions import walk_suffix


def main():

    parser = argparse.ArgumentParser(
        prog="MLClassification",
        description="Classifies audio data with already trained ml model",
    )

    parser.add_argument(
        "model_path",
        help="Path to trained model, e.g. /path/to/model/ml_training_gen1_resnet18_bypassFalse",
    )

    parser.add_argument(
        "folder_path",
        help="Folder path to audio files to classify",
    )

    parser.add_argument(
        "threshold",
        help="Decision threshold to round prediction scores",
        type=float,
    )

    parser.add_argument(
        "-w",
        "--workers",
        help="Number of workers for parallelization, i.e. cpus or cores. Default is 0 = current process.",
        default=0,
        type=int,
    )

    args = parser.parse_args()
    model_path = args.model_path
    folder_path = args.folder_path
    model_name = model_path.split(os.path.sep)[-1]

    mlmodel = walk_suffix(model_path, suffix=".model")  # TODO Koni
    with open(f"{model_path}/{model_name}_metadata.json") as file:
        model_metadata = json.load(file)

    original_df = create_audio_df(
        folder=folder_path, prediction_steps=float(model_metadata["sample_duration"])
    )
    os.makedirs(f"temp/{model_name}", exist_ok=True)

    # save original dataframe
    original_df.to_csv(
        f"./temp/{model_name}/original_df_{Path(folder_path).stem}_{model_name}.csv",
        index=True,
    )

    # CLassify audio data
    prediction_scores_df = prediction(
        model_path=next(mlmodel), df=original_df, workers=args.workers
    )

    prediction_scores_df.to_pickle(
        f"temp/{model_name}/prediction_df_{Path(folder_path).stem}_{model_name}.pkl"
    )
    prediction_scores_df.to_csv(
        f"temp/{model_name}/prediction_df_{Path(folder_path).stem}_{model_name}.csv",
        index=True,
    )

    print(
        f"Dataframe tables with prediction scores created and saved in ./temp/{model_name}."
    )

    classified_df = use_decision_threshold(
        df=prediction_scores_df, threshold=args.threshold
    )

    classified_df.to_pickle(
        f"temp/{model_name}/classified_df_{Path(folder_path).stem}_{model_name}.pkl"
    )
    classified_df.to_csv(
        f"temp/{model_name}/classified_df_{Path(folder_path).stem}_{model_name}.csv",
        index=True,
    )

    print(f"Classification dataframe tables created and saved in ./temp/{model_name}.")


if __name__ == "__main__":
    main()
