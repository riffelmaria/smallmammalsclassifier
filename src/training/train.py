### This training script trains an ML model for recognizing small terrestrial mammal calls in audio files ###

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from helper_functions.preprocess_functions import preprocess, set_conversion_table
from helper_functions.training_functions import (
    create_preprocessor,
    inspection_of_samples,
    train_model,
)
from lib.model_config import ModelConfig
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        prog="Training",
        description="Trains ML models with specified parameters",
    )

    parser.add_argument(
        "model_config",
        help="Path to JSON file of model config, e.g. src/lib/*.json",
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

    label_dfsegment = pd.DataFrame()
    label_dfs: list[pd.DataFrame] = []
    test_dfs_gen1 = []

    srs = []

    for jsonfile in Path(config.folder_jsons).iterdir():
        if "segments" in str(jsonfile):
            obj1, obj2 = preprocess(
                jsonfile, conversion_table, config.clip_sample_duration
            )
            label_dfsegment = obj1
            srs.append(obj2)
        else:
            obj1, obj2 = preprocess(
                jsonfile, conversion_table, config.clip_sample_duration
            )
            rest, test = train_test_split(obj1, test_size=0.1, shuffle=True)
            label_dfs.append(rest)
            test_dfs_gen1.append(test)
            srs.append(obj2)

    # Make separate test set for testing trained model
    test_df_gen1 = pd.concat(test_dfs_gen1)
    test_df_gen1 = test_df_gen1.replace(np.nan, 0)

    # preparation for stratifying label_df (collected data)
    segment_stratify = label_dfsegment.copy()
    segment_stratify["stratify_key"] = (
        label_dfsegment[["target_stm", "Noise", "bats"]]
        .astype(str)
        .agg("-".join, axis=1)
    )

    rest_segments, test_segments = train_test_split(
        label_dfsegment,
        test_size=0.1,
        shuffle=True,
        stratify=segment_stratify["stratify_key"],
    )

    # save test_df
    test_df = pd.concat([test_df_gen1, test_segments])
    os.makedirs(name=f"{args.output_path}/{config.model_name}", exist_ok=True)
    test_df.to_pickle(f"{args.output_path}/{config.model_name}/test_df.pkl")
    test_df.to_csv(f"{args.output_path}/{config.model_name}/test_df.csv", index=True)

    # Use train_size as "train_part" to train models with different number of samples

    for i, (train_df, validation_df) in enumerate(
        get_things(config, rest_segments, label_dfs)
    ):
        actual_training(train_df, validation_df, config, args, i)


def get_stratification(df: pd.DataFrame):
    df_stratify = df.copy()
    df_stratify["stratify_key"] = (
        df[["target_stm", "Noise", "bats"]].astype(str).agg("-".join, axis=1)
    )
    return df_stratify["stratify_key"]


def get_things(config, rest_segments, label_dfs):

    if config.train_size == 0.1:
        for _ in range(5):

            train_segments, _ = train_test_split(
                rest_segments,
                train_size=config.train_size,
                shuffle=True,
                stratify=get_stratification(rest_segments),
            )

            trainingsamples_df = pd.concat([train_segments, *label_dfs])
            trainingsamples_df = trainingsamples_df.replace(np.nan, 0)

            train_df, validation_df = train_test_split(
                trainingsamples_df,
                train_size=0.9,
                shuffle=True,
                stratify=get_stratification(trainingsamples_df),
            )
            yield train_df, validation_df

    elif config.train_size == 0.5:
        train_segments, _ = train_test_split(
            rest_segments,
            train_size=config.train_size,
            shuffle=True,
            stratify=get_stratification(rest_segments),
        )

        trainingsamples_df = pd.concat([train_segments, *label_dfs])
        trainingsamples_df = trainingsamples_df.replace(np.nan, 0)

        train_df, validation_df = train_test_split(
            trainingsamples_df,
            train_size=0.9,
            shuffle=True,
            stratify=get_stratification(trainingsamples_df),
        )
        yield train_df, validation_df
    elif config.train_size == 1.0:

        trainingsamples_df = pd.concat([rest_segments, *label_dfs])
        trainingsamples_df = trainingsamples_df.replace(np.nan, 0)

        train_df, validation_df = train_test_split(
            trainingsamples_df,
            train_size=0.9,
            shuffle=True,
            stratify=get_stratification(trainingsamples_df),
        )
        yield train_df, validation_df
    else:
        raise ValueError("help me, i do not belong here")


def actual_training(train_df, validation_df, config, args, i):
    preprocessor = create_preprocessor(
        config.clip_sample_duration,
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
        q = input("Continue?")
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


if __name__ == "__main__":
    main()
