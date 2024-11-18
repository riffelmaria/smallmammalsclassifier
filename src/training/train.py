### This training script trains an ML model for recognizing small terrestrial mammal calls in audio files ###

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from training_functions import (
    set_conversion_table,
    preprocess_dfs,
    create_preprocessor,
    inspection_of_samples,
    train_model,
    gradient_activation_cams # To do
)
from lib.model_config import ModelConfig
from sklearn.model_selection import train_test_split

from lib.genera import classes_gen
from lib.groups_minimal import classes_grp

def get_stratification(df: pd.DataFrame, scale: str):
    df_stratify = df.copy()
    class_scales = {"groups": classes_grp, "genera": classes_gen}
    if scale in class_scales:
        df_stratify["stratify_key"] = df[[class_scales[scale]]].astype(str).agg("-".join, axis=1)
    else:
        raise ValueError("Invalid scale")

    return df_stratify["stratify_key"]


def actual_training(train_df, validation_df, config, args, i):
    preprocessor = create_preprocessor(
        config.clip_duration,
        config.channels,
        config.min_fr,
        config.sampling_rate,
        width=config.imgsize,
        height=config.imgsize,
    )

    inspection_of_samples(
        train_df=train_df,
        n_samples=3,
        preprocessor=preprocessor,
        bypass_augmentations=config.bypass_augmentations,
        display_columns=config.display_columns,
        save_path="output/figures",
    )
    if args.ask:
        q = input("After inspection of samples: Continue? (type 'yes')")
        if q != "yes":
            return

    train_model(
        train_df,
        validation_df,
        architecture=config.architecture,
        clip_sample_duration=config.clip_sample_duration,
        preprocessor=preprocessor,
        entity=config.entity,
        project_name=config.project_name,
        group_name=config.group_name,
        model_name=f"{config.model_name}_run{i}",
        epochs_to_train=config.epochs_to_train,
        batch_size=config.batch_size,
        save_interval=config.save_interval,
        num_workers=config.num_workers,
        channels=config.channels,
        img_size=config.imgsize,
        output_path=args.output_path,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="MLTraining",
        description="Trains an ML model with specified parameters",
    )

    parser.add_argument(
        "model_config",
        type = str,
        help="Path to JSON file of model config, e.g. src/training/lib/*.json",
    )

    parser.add_argument(
        "--no-ask",
        action="store_false",
        dest="ask",
        help="flag, if no inspection of samples is needed",
    )
    parser.add_argument("output_path", help="Path to train dataset")
    args = parser.parse_args()

    with open(args.model_config) as file:
        arguments = json.load(file)
        config = ModelConfig(**arguments)

    conversion_table = set_conversion_table(config.scale)

    # Preprocess files to data frames
    #label_dfsegment = pd.DataFrame()
    label_dfs: list[pd.DataFrame] = []
    test_dfs: list[pd.DataFrame] = []

    srs = []

    for jsonfile in Path(config.folder_jsons).iterdir():
        obj1, obj2 = preprocess_dfs(
                jsonfile, 
                conversion_table, 
                config.clip_duration
            )
        rest, test = train_test_split(obj1, test_size=0.1, shuffle=True)
        label_dfs.append(rest)
        test_dfs.append(test)
        srs.append(obj2)

    # Make separate test set for testing trained model
    test_dataset_df = pd.concat(test_dfs)
    test_dataset_df = test_dataset_df.replace(np.nan, 0)
    os.makedirs(name=f"{args.output_path}/{config.model_name}", exist_ok=True)
    test_dataset_df.to_pickle(f"{args.output_path}/{config.model_name}/test_dataset_df.pkl")
    test_dataset_df.to_csv(f"{args.output_path}/{config.model_name}/test_dataset_df.csv", index=True)

    # Ensure that `label_df` is stratified by class so that each class is represented in both the training and validation datasets
    label_df = pd.concat(label_dfs)
    label_df = label_df.replace(np.nan, 0)
    stratify_key = get_stratification(df = label_df, scale =config.scale)

    train_df, validation_df = train_test_split(
        label_df,
        train_size=config.train_size,
        shuffle=True,
        stratify=stratify_key,
    )

    actual_training(train_df, validation_df, config, args)



if __name__ == "__main__":
    main()
