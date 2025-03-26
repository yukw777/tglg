from pathlib import Path

from jsonargparse import auto_cli
from torch.utils.data import Dataset
from tqdm import tqdm

# decord must be imported after torch
# https://github.com/dmlc/decord/issues/293
from decord import VideoReader  # isort: skip
import json


def main(real_time_dataset: Dataset, out_file: Path) -> None:
    videos = set(
        (datapoint["video_id"], datapoint["video_path"])
        for datapoint in iter(real_time_dataset)
    )
    video_stats = {}
    for video_id, video_path in tqdm(videos, desc="Obtain Video Stats"):
        vr = VideoReader(str(video_path))
        video_stats[video_id] = {"fps": vr.get_avg_fps(), "num_frames": len(vr)}

    with open(out_file, "w") as f:
        json.dump(video_stats, f, indent=4)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
