import pytest
import torch

from real_time_vlm_benchmark.baseline_models.videollm_online_models.videollm_online import (
    construct_interleaved_dialogue,
)


@pytest.mark.parametrize(
    "dialogue,frame_timestamps,expected",
    [
        (
            [
                {"role": "assistant", "eval": False, "start": 3},
                {"role": "assistant", "eval": False, "start": 5},
                {"role": "assistant", "eval": True, "start": 8},
            ],
            torch.arange(0, 10, 0.5).tolist(),
            (
                [
                    {"role": "system", "content": "system message"},
                    {"role": "stream", "num_frames": 7, "learn": False},
                    {"role": "assistant", "eval": False, "start": 3},
                    {"role": "stream", "num_frames": 4, "learn": False},
                    {"role": "assistant", "eval": False, "start": 5},
                ],
                11,
            ),
        ),
        (
            [
                {"role": "assistant", "eval": False, "start": 3},
                {"role": "assistant", "eval": False, "start": 5},
                {"role": "assistant", "eval": True, "start": 8},
            ],
            torch.arange(2, 10, 0.5).tolist(),
            (
                [
                    {"role": "system", "content": "system message"},
                    {"role": "stream", "num_frames": 3, "learn": False},
                    {"role": "assistant", "eval": False, "start": 3},
                    {"role": "stream", "num_frames": 4, "learn": False},
                    {"role": "assistant", "eval": False, "start": 5},
                ],
                7,
            ),
        ),
        (
            [
                {"role": "assistant", "eval": False, "start": 3},
                {"role": "assistant", "eval": False, "start": 5},
                {"role": "assistant", "eval": True, "start": 8},
            ],
            torch.arange(4, 10, 0.5).tolist(),
            (
                [
                    {"role": "system", "content": "system message"},
                    {"role": "stream", "num_frames": 3, "learn": False},
                    {"role": "assistant", "eval": False, "start": 5},
                ],
                3,
            ),
        ),
        (
            [
                {"role": "assistant", "eval": False, "start": 3},
                {"role": "assistant", "eval": False, "start": 5},
                {"role": "user", "eval": False, "start": 6},
                {"role": "assistant", "eval": False, "start": 8},
                {"role": "assistant", "eval": True, "start": 10},
                {"role": "assistant", "eval": True, "start": 12},
            ],
            torch.arange(0, 10, 0.25).tolist(),
            (
                [
                    {"role": "system", "content": "system message"},
                    {"role": "stream", "num_frames": 13, "learn": False},
                    {"role": "assistant", "eval": False, "start": 3},
                    {"role": "stream", "num_frames": 8, "learn": False},
                    {"role": "assistant", "eval": False, "start": 5},
                    {"role": "stream", "num_frames": 4, "learn": False},
                    {"role": "user", "eval": False, "start": 6},
                    {"role": "stream", "num_frames": 8, "learn": False},
                    {"role": "assistant", "eval": False, "start": 8},
                ],
                33,
            ),
        ),
        (
            [
                {"role": "assistant", "eval": False, "start": 3},
                {"role": "assistant", "eval": False, "start": 5},
                {"role": "user", "eval": False, "start": 6},
                {"role": "assistant", "eval": False, "start": 8},
                {"role": "assistant", "eval": True, "start": 10},
                {"role": "assistant", "eval": True, "start": 12},
            ],
            torch.arange(5.5, 10, 0.25).tolist(),
            (
                [
                    {"role": "system", "content": "system message"},
                    {"role": "stream", "num_frames": 3, "learn": False},
                    {"role": "user", "eval": False, "start": 6},
                    {"role": "stream", "num_frames": 8, "learn": False},
                    {"role": "assistant", "eval": False, "start": 8},
                ],
                11,
            ),
        ),
    ],
)
def test_construct_interleaved_dialogue(
    dialogue: list[dict],
    frame_timestamps: list[float],
    expected: list[dict],
) -> None:
    assert (
        construct_interleaved_dialogue(
            dialogue, frame_timestamps, lambda x: "system message"
        )
        == expected
    )
