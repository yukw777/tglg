from collections import defaultdict
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from real_time_vlm_benchmark.baseline_models.utils.sample import (
    sample_frames_for_dialogue,
)


def convert_to_frame_dataset(
    real_time_dataset: Dataset,
    sample_fps: float,
    max_num_frames: int | None = None,
    show_progress: bool = True,
) -> list[dict]:
    frame_data: dict[tuple[str, Path], set[int]] = defaultdict(set)
    for datapoint in tqdm(
        real_time_dataset, disable=not show_progress, desc="Convert to Frame Dataset"
    ):
        dialogue = datapoint["dialogue"]

        vidcap = cv2.VideoCapture(str(datapoint["video_path"]))
        frame_idx, _, _ = sample_frames_for_dialogue(
            dialogue,
            vidcap.get(cv2.CAP_PROP_FPS),
            sample_fps,
            max_num_frames=max_num_frames,
        )
        for frame_id in frame_idx.tolist():
            frame_data[(datapoint["video_id"], datapoint["video_path"])].add(frame_id)

    return [
        {
            "video_id": video_id,
            "video_path": video_path,
            "frame_idx": torch.tensor(sorted(idx)),
        }
        for (video_id, video_path), idx in frame_data.items()
    ]
