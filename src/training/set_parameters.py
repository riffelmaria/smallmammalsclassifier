import argparse
import json
import os

from pydantic import ValidationError
from shared.model_config import ModelConfig

default_config = ModelConfig(
    entity="-",  # wandb parameter
    project_name="-",  # wandb parameter
    model_name="test",
    run_name="-",  # wandb parameter
    group_name="-",  # wandb parameter
    epochs_to_train=100,
    save_interval=1,  # save every epoch
    num_workers=0,  # define number of kernels to use for training; 0 = root process
    train_size=0.7,  # proportion to split data in training and validation datasets
    clip_duration=1.2,
    n_samples=3,
    channels=3,
    min_fr=0,
    display_columns=3,  # parameter for sample inspection
    batch_size=32,  # "Rule of thumb", a higher bathc size like 64 or 128 ... to do
    bypass_augmentations=False,  # add augmentation to samples
    scale="groups",
    folder_jsons="src/training/lib/",
    architecture="resnet18",
    imgsize=448,
    sampling_rate=192000,
)


def get_user_parameters(cfg: dict | None = None) -> ModelConfig:
    """
    Prompt the user to accept or modify default parameters for model config.

    Args:
        cfg (dict, optional):
            A dictionary containing the initial configuration parameters.
            If None, a default configuration will be used.

    Returns:
        dict: Dictionary of parameters with updated values as entered by the user.
    """
    if cfg is None:
        updated_params = default_config.copy()
    else:
        updated_params = cfg.copy()

    print("Current parameters:")
    for param, value in updated_params.items():
        print(f"  {param}: {value}")

    accept = input("Do you accept the current parameters? (yes/no): ").strip().lower()

    if accept == "yes":
        print("Using current parameters.")
        return updated_params
    elif accept != "no":
        print("Invalid input. Please respond with 'yes' or 'no'.")
        return get_user_parameters(updated_params)

    print("Please enter new values for each parameter:")

    for param in updated_params:
        new_value = input(f"Enter value for {param} [{updated_params[param]}]: ")
        if new_value != "":
            updated_params[param] = new_value
    try:
        # Validate and create a new Parameters object
        validated_params = ModelConfig(**updated_params)
        return validated_params
    except ValidationError as e:
        print("Invalid input. Please try again.")
        print(e)
        return get_user_parameters(updated_params)


def main():
    parser = argparse.ArgumentParser(
        prog="SetParameters",
        description="Writes a JSON file with the defined model parameters",
    )

    parser.add_argument(
        "-f",
        "--folder_name",
        help="Optional, path to folder where to store the model parameter JSON",
        default="training/lib",
    )
    args = parser.parse_args()

    model_config = get_user_parameters()

    os.makedirs(name=args.folder_name, exist_ok=True)

    json_object = json.dumps(model_config.__dict__)
    with open(
        args.folder_name + "/" + model_config.model_name + ".json", "w"
    ) as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    main()
