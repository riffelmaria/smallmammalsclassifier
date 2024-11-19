import json
import os
from typing import Callable, List

from shared.audio_functions import walk_wav


def dict_to_jsonfile(dictionary, folder_out, filename):
    json_object = json.dumps(dictionary, indent=4)

    # Writing to .json
    with open(folder_out + "/" + filename, "w") as outfile:
        outfile.write(json_object)


def find_files(folder: str, walk_func: Callable, suffix: str | None = None) -> List:
    parentfolder = os.path.abspath(folder)

    audio_files = [
        file
        for file in walk_func(parentfolder)
        if suffix is None or os.path.exists(f"{os.path.splitext(file)[0]}{suffix}")
    ]
    return audio_files
