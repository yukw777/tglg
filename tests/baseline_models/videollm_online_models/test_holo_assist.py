import pytest

from real_time_vlm_benchmark.baseline_models.videollm_online_models.holo_assist import (
    construct_interleaved_dialogue,
    sample_frames_for_dialogue,
)


@pytest.mark.parametrize(
    "start_time,end_time,video_avg_fps,sample_fps,expected_idx",
    [
        (0, 5, 2, 2, list(range(0, 11, 1))),
        (3, 8, 2, 2, list(range(6, 17, 1))),
        (0, 5, 4, 2, list(range(0, 21, 2))),
        (3, 8, 4, 2, list(range(12, 33, 2))),
    ],
)
def test_sample_frames_for_dialogue(
    start_time: float,
    end_time: float,
    video_avg_fps: float,
    sample_fps: float,
    expected_idx: list[int],
) -> None:
    assert (
        sample_frames_for_dialogue(
            start_time, end_time, video_avg_fps, sample_fps
        ).tolist()
        == expected_idx
    )


@pytest.mark.parametrize(
    "dialogue,sample_fps,use_narration,expected",
    [
        (
            [
                {"role": "assistant", "eval": False, "start": 3},
                {"role": "assistant", "eval": False, "start": 5},
                {"role": "assistant", "eval": True, "start": 8},
            ],
            2,
            False,
            (
                [
                    {
                        "role": "system",
                        "content": "A multimodal AI assistant is helping users with some activities. Below is their conversation, interleaved with the list of video frames received by the assistant.",
                    },
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
                {"role": "user", "eval": False, "start": 6},
                {"role": "assistant", "eval": False, "start": 8},
                {"role": "assistant", "eval": True, "start": 10},
                {"role": "assistant", "eval": True, "start": 12},
            ],
            4,
            False,
            (
                [
                    {
                        "role": "system",
                        "content": "A multimodal AI assistant is helping users with some activities. Below is their conversation, interleaved with the list of video frames received by the assistant.",
                    },
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
                {"role": "system", "eval": False, "content": "summary", "start": 0},
                {"role": "assistant", "eval": False, "start": 3},
                {"role": "assistant", "eval": False, "start": 5},
                {"role": "assistant", "eval": True, "start": 8},
            ],
            2,
            True,
            (
                [
                    {
                        "role": "system",
                        "content": (
                            "A multimodal AI assistant is helping users with some activities. "
                            "Below is their conversation, interleaved with the list of video frames received by the assistant. "
                            "The assistant should give the user instructions and correct their mistakes. "
                            "Here's the summary of the activity: summary"
                        ),
                    },
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
                {"role": "system", "eval": False, "content": "summary", "start": 0},
                {"role": "assistant", "eval": False, "start": 3},
                {"role": "assistant", "eval": False, "start": 5},
                {"role": "user", "eval": False, "start": 6},
                {"role": "assistant", "eval": False, "start": 8},
                {"role": "assistant", "eval": True, "start": 10},
                {"role": "assistant", "eval": True, "start": 12},
            ],
            4,
            True,
            (
                [
                    {
                        "role": "system",
                        "content": (
                            "A multimodal AI assistant is helping users with some activities. "
                            "Below is their conversation, interleaved with the list of video frames received by the assistant. "
                            "The assistant should give the user instructions and correct their mistakes. "
                            "Here's the summary of the activity: summary"
                        ),
                    },
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
    ],
)
def test_construct_interleaved_dialogue(
    dialogue: list[dict], sample_fps: float, use_narration: bool, expected: list[dict]
) -> None:
    assert (
        construct_interleaved_dialogue(
            dialogue, sample_fps, use_narration=use_narration
        )
        == expected
    )
