from pathlib import Path

import typing
#import opensoundscape.ml.cnn
import wandb
from matplotlib import pyplot as plt
import pandas as pd
from shared.audio_functions import get_audiofiles_from_jsonfile
from typing import Tuple
from opensoundscape import Audio
from opensoundscape import CNN, SpectrogramPreprocessor
from opensoundscape.annotations import BoxedAnnotations
from opensoundscape.ml.cnn import load_model
from opensoundscape.ml.datasets import AudioFileDataset
from opensoundscape.ml.loss import BCEWithLogitsLoss_hot
from opensoundscape.preprocess.utils import show_tensor_grid

from lib.genera import genera
from lib.groups_minimal import groups


def set_conversion_table(class_scale: str):
    conversion_table = dict()
    if class_scale == "groups":
        conversion_table = groups
    elif class_scale == "genera":
        conversion_table = genera
    else:
        raise ValueError

    return conversion_table


def preprocess_dfs(
    jsonfile: Path, 
    conversion_table: dict, 
    clip_duration: float
) -> Tuple[pd.DataFrame, int]:
    """Preprocessing audio files and raven annotation files

    Make a dataframe object from audio files and raven annotations files.
    Define all preprocessing parameters that determine training and prediciton performance.
    """

    # Make an object from raven files
    audio_files, raven_files = get_audiofiles_from_jsonfile(jsonfile)
    boxed_annotations = BoxedAnnotations.from_raven_files(
        raven_files=raven_files,
        audio_files=audio_files,
        annotation_column_name="Annotation",
    )

    # Use conversion table for the annotation column
    boxed_annotations_converted = boxed_annotations.convert_labels(
        conversion_table=conversion_table
    )

    # Create data frame with clips from the annotated audio files
    label_df = boxed_annotations_converted.one_hot_clip_labels(
        clip_duration=clip_duration,
        clip_overlap=clip_duration * 0.4,  # 40% overlap of consecutive clips
        min_label_overlap=clip_duration * 0.3,  # minimum of 30% label overlap within a clip
        class_subset=None,
        final_clip="full", # last clip in an audio file that is not long enough will be 
        audio_files=list(boxed_annotations_converted.df["audio_file"].unique()),
    )

    # get sampling rate from audio files
    sr = min([Audio.from_file(file).sample_rate for file in audio_files])

    return typing.cast(pd.DataFrame, label_df), sr


def create_preprocessor(
    clip_duration: float, 
    channels: int, 
    min_fr: int, 
    sr: int, 
    width: int=448, 
    height: int=448
):
    preprocessor = SpectrogramPreprocessor(
        sample_duration=clip_duration,
        channels=channels,
        width=width,
        height=height,
    )
    preprocessor.pipeline.bandpass.set(min_f=min_fr, max_f=sr / 2)

    return preprocessor


def inspection_of_samples(
    *,
    train_df,
    n_samples: int,
    preprocessor,
    bypass_augmentations: bool = True,
    display_columns: int = 3,
    save_path: str = "../../sample_inspection",
):
    """Inspection of spectrogram samples

    For trouble-shooting before model training.
    Shows how the input of spectrogram samples look like.
    """
    sample_of_n = train_df.sample(n=n_samples)
    inspection_dataset = AudioFileDataset(sample_of_n, preprocessor)
    inspection_dataset.bypass_augmentations = bypass_augmentations

    samples = [sample.data for sample in inspection_dataset]
    labels = [
        list(sample.labels[sample.labels > 0].index) for sample in inspection_dataset
    ]
    # display the samples
    fig = show_tensor_grid(samples, columns=display_columns, labels=labels)
    fig.savefig(f"{save_path}/inspection.png")


def train_model(
    train_df,
    validation_df,
    architecture: str,
    clip_duration: float,
    preprocessor,
    epochs_to_train: int,
    batch_size: int,
    save_interval: int,
    num_workers: int,
    img_size: int,
    channels: int,
    output_path:str = "../../binary_train",
    model_name: str = "",
    wandb_usage: bool = 0,
    entity: str = "",
    project_name: str = "",
    group_name: str = "",
):
    classes = train_df.columns

    model = CNN(
        architecture,
        classes=classes,
        sample_duration=clip_duration,
        sample_shape=(img_size, img_size, channels),
    )

    BCEWithLogitsLoss_hot() #change loss-function if needed

    model.preprocessor = preprocessor

    if wandb_usage:
        try:
            wandb.login()
            wandb_session = wandb.init(
                                entity = entity,
                                project = project_name, 
                                group = group_name, 
                                name = model_name
                            )
            # optional: wandb.log; determines how model flow data will be transfered to wandb (as a dictionary)
            # Further dependent variables like loss and accuracy or output metrics should be saved with wandb.log.
        except RuntimeError as rte:
            print(rte)

        except:
            print("Failed to create wandb session. wandb session will be None\n")
            wandb_session = None

    model.train(
        train_df = train_df,
        validation_df = validation_df,
        save_path = f"{output_path}/{model_name}/",  # where to save the trained model
        epochs = epochs_to_train,
        batch_size = batch_size,  # larger batch sizes (64+) improve stability and generalizability of training especially for ResNet
        save_interval = save_interval,  # save model every epoch (the best model is always saved in addition)
        # invalid_samples_log = "./invalid_samples_log/" + model_name,
        num_workers = num_workers,  # specify 4 if you have 4 CPU processes, e.g. 0 means only the root process
        wandb_session = wandb_session
    )

    if wandb_usage:
        wandb.unwatch(model.network)
        wandb.finish()  # so that wandb knows the training finished successfully

    print("Model successfully trained.\n")


def gradient_activation_cams(model_path, train_df, output_path="./output/figures/"):
    """Check gradient activation maps

    For trouble-shooting after model training. 
    Shows which part of the spectrogram is used to recognize the sound.
    """

    model = load_model(model_path)
    model_name = Path(model_path).parent.name

    samples = model.generate_cams(samples=train_df.head(1))
    samples[0].cam.plot()

    # Plot loss history
    plt.scatter(model.loss_hist.keys(), model.loss_hist.values())
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(f"{output_path}/GAM_{model_name}.png")
    plt.show()
    plt.close()
