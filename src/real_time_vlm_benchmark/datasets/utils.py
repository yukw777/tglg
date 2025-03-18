from collections import defaultdict
from copy import deepcopy

import torch
from torch.utils.data import Dataset

# decord must be imported after torch
# https://github.com/dmlc/decord/issues/293
from decord import VideoReader  # isort: skip
from pathlib import Path

from tqdm import tqdm

from real_time_vlm_benchmark.baseline_models.utils.sample import (
    sample_frames_for_dialogue,
)


def convert_real_time_anns_to_datapoint(
    anns: dict[str, dict],
) -> list[tuple[str, list[dict]]]:
    data: list[tuple[str, list[dict]]] = []
    for video, dialogue in anns.items():
        i = 0
        is_eval = False
        while i < len(dialogue):
            if not is_eval and dialogue[i]["eval"]:
                is_eval = True
            if is_eval and not dialogue[i]["eval"]:
                is_eval = False
                data.append((video, deepcopy(dialogue[:i])))
                # set eval to False for the added utterances
                # as they will be used as part of the context for the next data point.
                for utter in dialogue[:i]:
                    utter["eval"] = False
            i += 1
        # take care of the stragglers
        if is_eval:
            data.append((video, deepcopy(dialogue[:i])))
    return data


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
        vr = VideoReader(str(datapoint["video_path"]))
        dialogue = datapoint["dialogue"]

        frame_idx, _, _ = sample_frames_for_dialogue(
            dialogue,
            vr.get_avg_fps(),
            sample_fps,
            len(vr),
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
