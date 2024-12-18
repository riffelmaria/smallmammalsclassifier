# this script will create a jsonfile from audio and raven files to be used in the preprocessing function of the training
import argparse
import os
from pathlib import Path

from shared.audio_functions import walk_wav
from shared.files_manipulation_functions import dict_to_jsonfile


def write_dictionary(audio_files: list, suffix: str):
    dictionary = [
        dict(
            audio_file=x,
            raven_file=os.path.splitext(x)[0] + suffix,
        )
        for x in audio_files
    ]
    return dictionary


def find_files(folder, suffix):
    parentfolder = os.path.abspath(folder)

    audio_files = [
        file
        for file in walk_wav(parentfolder)
        if os.path.exists(os.path.splitext(file)[0] + suffix)
    ]
    return audio_files


def write_json_files(
    folder_path: str, folder_out: str, suffix: str
) -> None:  # TODO suffix der raven files!
    audio_files = find_files(folder_path, suffix)
    dictionary = write_dictionary(audio_files=audio_files, suffix=suffix)
    dict_to_jsonfile(
        dictionary=dictionary,
        folder_out=folder_out,
        filename=f"training_data_{Path(folder_path).stem}.json",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="JSONWriter",
        description="Writes JSON files with file names of audio data with Raven tables to be used for model trianing",
    )

    parser.add_argument(
        "folder_in",
        help="Folder path to audio data and Raven tables, see project description for data structure example",
        type=str,
    )

    parser.add_argument(
        "-f",
        "--folder_out",
        help="Folder path where to write the json files (should be ./src/training/lib)",
        default="./src/training/lib",
        type=str,
    )

    parser.add_argument(
        "-s",
        "--suffix",
        help="Common suffix of all Raven files created for the training audio files, e.g. *.Table.1.selections.txt",
        default=".Table.1.selections.txt",
        type=str,
    )

    args = parser.parse_args()

    write_json_files(
        folder_path=args.folder_in,
        folder_out=args.folder_out,
        suffix=args.suffix,
    )

    print("JSON file created from audio and Raven file names.")


if __name__ == "__main__":
    main()
