# this script will create a jsonfile from audio and raven files to be used in the preprocessing function of the training
import argparse
import json
import os
from glob import glob

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


def write_json_files(folders: dict, folder_out, suffix: str):

    for name, folder in folders.items():
        # Find all audio files with corresponding raven files
        audio_files = find_files(folder, suffix=suffix)

        # Data to be written
        dictionary = write_dictionary(audio_files, suffix=suffix)
        dict_to_jsonfile(dictionary, folder_out, name)


def main():
    parser = argparse.ArgumentParser(
        prog="JSONWriter",
        description="Writes JSON files with file names of audio data to be classified by a model",
    )

    parser.add_argument(
        "folder_in",
        help="folder path to audio data",
    )

    parser.add_argument(
        "folder_out",
        help="folder path where to write the json files to (should be in git/data/)",  # git/temp?
    )
    args = parser.parse_args()

    folder_in = args.folder_in
    folder_out = "./temp/"

    # walk folder_in
    for audiofile in walk_wav(folder_in):
        ...

    write_json_files(
        folders_in,
        args.folder_out + "gen2_files_jsons",
        suffix=".Table.1.selections.txt",
    )


if __name__ == "__main__":
    main()
