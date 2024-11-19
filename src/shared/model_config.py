from pydantic.dataclasses import dataclass


@dataclass
class ModelConfig:
    entity: str
    project_name: str
    model_name: str
    run_name: str
    group_name: str
    epochs_to_train: int
    save_interval: int
    num_workers: int
    train_size: float
    clip_duration: float
    n_samples: int
    channels: int
    min_fr: int
    display_columns: int
    batch_size: int
    bypass_augmentations: bool
    scale: str
    folder_jsons: str
    architecture: str
    imgsize: int
    sampling_rate: int
