import pytest

from real_time_vlm_benchmark.baseline_models.utils.sample import (
    sample_frames_for_dialogue,
)


@pytest.mark.parametrize(
    "dialogue,video_avg_fps,sample_fps,video_num_frames,max_num_frames,expected_idx,expected_start_time,expected_end_time",
    [
        (
            [
                {"role": "system"},
                {"role": "user", "start": 0},
                {"role": "assistant", "end": 5},
            ],
            2,
            2,
            11,
            None,
            list(range(0, 11, 1)),
            0,
            5,
        ),
        (
            [
                {"role": "system"},
                {"role": "user", "start": 2},
                {"role": "assistant", "end": 5},
            ],
            2,
            2,
            11,
            None,
            list(range(4, 11, 1)),
            2,
            5,
        ),
        (
            [
                {"role": "system"},
                {"role": "user", "start": 0},
                {"role": "assistant", "end": 8},
            ],
            2,
            2,
            17,
            10,
            list(range(6, 17, 1)),
            3,
            8,
        ),
        (
            [
                {"role": "system"},
                {"role": "user", "start": 2},
                {"role": "assistant", "end": 10},
            ],
            2,
            2,
            21,
            10,
            list(range(10, 21, 1)),
            5,
            10,
        ),
        (
            [
                {"role": "system"},
                {"role": "user", "start": 0},
                {"role": "assistant", "end": 5},
            ],
            4,
            2,
            21,
            None,
            list(range(0, 21, 2)),
            0,
            5,
        ),
        (
            [
                {"role": "system"},
                {"role": "user", "start": 0},
                {"role": "assistant", "end": 8},
            ],
            4,
            2,
            33,
            10,
            list(range(12, 33, 2)),
            3,
            8,
        ),
        (
            [
                {"role": "system"},
                {"role": "user", "start": 0},
                {"role": "assistant", "end": 8},
            ],
            4.2,
            2,
            33,
            10,
            [12, 14, 16, 18, 20, 23, 25, 27, 29, 32],
            3,
            8,
        ),
    ],
)
def test_sample_frames_for_dialogue(
    dialogue: list[dict],
    video_avg_fps: float,
    sample_fps: float,
    video_num_frames: int,
    max_num_frames: int | None,
    expected_idx: list[int],
    expected_start_time: float,
    expected_end_time: float,
) -> None:
    frame_idx, start_time, end_time = sample_frames_for_dialogue(
        dialogue,
        video_avg_fps,
        sample_fps,
        video_num_frames,
        max_num_frames=max_num_frames,
    )
    assert (frame_idx.tolist(), start_time, end_time) == (
        expected_idx,
        expected_start_time,
        expected_end_time,
    )
