import argparse
import csv
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import soundfile as sf
from scipy.io import wavfile
from shared.audio_functions import walk_wav


def parse_mm_ss(value: str) -> timedelta:
    """
    Parse a string in the format mm:ss into a timedelta object.

    Args:
        value (str): A string representing a time duration in mm:ss format.

    Returns:
        timedelta: A timedelta object representing the duration.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be parsed.
    """
    try:
        minutes, seconds = map(int, value.split(":"))
        return timedelta(minutes=minutes, seconds=seconds)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Time must be in the format mm:ss (e.g., 05:30)."
        )


def write_audiosegments(audio_file, duration, new_filename) -> None:
    """
    Args:
        duration (int): Duration of a segment/ audio snippet in seconds.

    """
    if not os.path.exists(audio_file):
        raise ValueError("Filepath does not exist.")

    sr, audio = wavfile.read(audio_file, mmap=True)
    audio_length = len(audio)
    for start in range(0, audio_length, duration * sr):
        start_frame = int(start)
        end_frame = int(start + duration * sr)
        if end_frame >= audio_length:
            end_frame = audio_length - 1
        segment = audio[start_frame:end_frame]
        sf.write(
            new_filename.format(start=int(start_frame / sr), end=int(end_frame / sr)),
            segment,
            sr,
        )
    print(f"Number of snippet files: {len(range(0, audio_length, duration*sr))}")


def process_folder(
    folder_path: str, folder_to_snippets: str, duration: timedelta
) -> None:
    for audiofile in walk_wav(folder_path, ignore_path=folder_to_snippets):
        write_audiosegments(
            audio_file=audiofile,
            duration=int(duration.total_seconds()),
            new_filename=f"{folder_to_snippets}/{Path(audiofile).parent.stem}_{Path(audiofile).stem}"
            + "_start{start}s_end{end}s.WAV",
        )


def main():
    parser = argparse.ArgumentParser(
        prog="AudioSnippetWriter",
        description="Writes audio snippet/segment files and their corresponding raven files.",
    )

    parser.add_argument(
        "base_path",
        help="Base path of the folder where the audio files exist",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--snippet_folder",
        help="Optional, folder path where to store the created audio snippets",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--duration",
        help="Optional, audio snippet duration in mm:ss format, e.g. 05:30 for 5 minutes and 30 seconds",
        default="05:00",
        type=parse_mm_ss,
    )

    args = parser.parse_args()
    base_path = args.base_path

    if args.snippet_folder is None:
        snippet_folder = f"{base_path}/Audio_Snippets"
    else:
        snippet_folder = args.snippet_folder
    os.makedirs(name=snippet_folder, exist_ok=True)

    process_folder(base_path, snippet_folder, args.duration)

    print("All audio files processed.")


if __name__ == "__main__":
    main()
