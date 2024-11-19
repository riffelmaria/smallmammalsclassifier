import csv
import json
import os
from pathlib import Path
from typing import Generator, Iterable

import exiftool
import soundfile as sf
from scipy.io import wavfile


def get_file_metadata(
    file_path, params=["-FileModifyDate", "-Duration", "-SampleRate", "-ext wav"]
):  # TODO: typing
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(
            file_path,
            params=params,
        )
    return metadata[0]


def write_raventable_from_audiofilename(
    audio_filename: str, rows: Iterable, suffix=".Table.2.selections.txt"
):
    return write_raventable(get_raven_filename_from_audio(audio_filename, suffix), rows)


def write_raventable(raven_filename, rows: Iterable):
    fieldnames = [
        "Selection",
        "View",
        "Channel",
        "Begin Time (s)",
        "End Time (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Delta Time (s)",
        "Delta Freq (Hz)",
        "Avg Power Density (dB FS/Hz)",
        "Annotation",
    ]

    with open(raven_filename, "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, dialect="excel-tab")
        writer.writeheader()
        writer.writerows(rows)


def get_raven_filename_from_audio(audio_file, suffix=".Table.2.selections.txt") -> str:
    return audio_file[: -len(".wav")] + suffix


def get_audio_filename_from_raven(
    raven_file, raven_suffix=".Table.2.selections.txt"
) -> str:
    return raven_file[: -len(raven_suffix)] + ".WAV"


def get_audiofiles_from_jsonfile(jsonfile: Path):
    with open(jsonfile) as file:
        files = json.load(file)

    audio_files = [x.get("audio_file") for x in files]
    raven_files = [x.get("raven_file") for x in files]

    return audio_files, raven_files


def write_audiosegment(audio_file, start, end, target_filename) -> None:
    print(audio_file, start, end, target_filename)

    if not os.path.exists(audio_file):
        raise ValueError("Filepath does not exist.")

    sr, y = wavfile.read(audio_file, mmap=True)
    start_frame = int(start * sr)
    end_frame = int(end * sr)
    segment = y[start_frame:end_frame]

    sf.write(target_filename, segment, sr)


def get_target_filename(
    folder_to_segments,
    audiofile_path,
    **kwargs,
) -> str:
    kwargs = {**kwargs}

    kwargs_str = "_".join("{}{}".format(key, value) for key, value in kwargs.items())

    pathpart = "__".join(audiofile_path.split(os.path.sep)[5:])[: -len(".WAV")]
    return f"{folder_to_segments}/{pathpart}__{kwargs_str}.WAV"


def get_annotation_files(
    folder_path: str, suffix=".Table.2.selections.txt"
) -> Generator:
    for root, dirs, files in os.walk(folder_path):
        for raven_file in files:
            if not raven_file.endswith(suffix):
                continue
            yield os.path.join(root, raven_file)


def walk_mp4(folder_path: str):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".mp4")):
                yield os.path.join(root, file)


def walk_wav(folder_path: str, ignore_path: str | None = None):
    for root, dirs, files in os.walk(folder_path):
        if root == ignore_path:
            continue
        for file in files:
            if file.lower().endswith((".wav")):
                yield os.path.join(root, file)


def walk_suffix(folder_path: str, suffix: str = "Table.2.selections.txt"):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                yield os.path.join(root, file)
