import math
from queue import Queue
from typing import Any

import torch
from torch.utils.data import Sampler


def sample_frames_for_dialogue(
    dialogue: list[dict],
    video_avg_fps: float,
    sample_fps: float,
    video_num_frames: int,
    max_num_frames: int | None = None,
) -> tuple[torch.Tensor, float, float]:
    """
    Returns the indices for frames sampled at the given fps for the given dialogue.
    """
    # sample frames from max(0, end time - max_num_frames/sample_fps) to the end time of the last utterance at self.frame_fps
    end_time = dialogue[-1]["end"]
    if max_num_frames is None:
        start_time = 0
    else:
        start_time = max(0, end_time - max_num_frames / sample_fps)
    start_time_frame = math.ceil(start_time * video_avg_fps)
    end_time_frame = min(math.floor(end_time * video_avg_fps), video_num_frames - 1)
    num_frames = end_time_frame - start_time_frame + 1
    frame_interval = video_avg_fps / sample_fps
    num_frames_to_sample = math.ceil(num_frames / frame_interval)
    return (
        torch.linspace(start_time_frame, end_time_frame, num_frames_to_sample).to(
            torch.int
        ),
        start_time,
        end_time,
    )


class QueueSampler(Sampler):
    def __init__(self, queue: Queue) -> None:
        self.queue = queue

    def __iter__(self) -> Any:
        while not self.queue.empty():
            yield self.queue.get()

    def __len__(self) -> int:
        return self.queue.qsize()
