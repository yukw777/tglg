import pytest

from real_time_vlm_benchmark.baseline_models.utils.sample import (
    sample_frames_for_dialogue,
)


@pytest.mark.parametrize(
    "dialogue,video_avg_fps,sample_fps,max_num_frames,expected_idx,expected_start_time,expected_end_time",
    [
        (
            [
                {"role": "system"},
                {"role": "user", "start": 0},
                {"role": "assistant", "end": 5},
            ],
            2,
            2,
            None,
            list(range(0, 10, 1)),
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
            None,
            list(range(4, 10, 1)),
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
            10,
            list(range(6, 16, 1)),
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
            10,
            list(range(10, 20, 1)),
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
            None,
            list(range(0, 17, 2)) + [19],
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
            10,
            list(range(12, 29, 2)) + [31],
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
    max_num_frames: int | None,
    expected_idx: list[int],
    expected_start_time: float,
    expected_end_time: float,
) -> None:
    frame_idx, start_time, end_time = sample_frames_for_dialogue(
        dialogue,
        video_avg_fps,
        sample_fps,
        max_num_frames=max_num_frames,
    )
    assert (frame_idx.tolist(), start_time, end_time) == (
        expected_idx,
        expected_start_time,
        expected_end_time,
    )
