import pytest

from real_time_vlm_benchmark.baseline_models.videollm_online_models.holo_assist import (
    _construct_system_message,
)


@pytest.mark.parametrize(
    "use_narration,expected",
    [
        (
            True,
            (
                "A multimodal AI assistant is helping users with some activities. "
                "Below is their conversation, interleaved with the list of video frames received by the assistant. "
                "The assistant should give the user instructions and correct their mistakes. "
                "Here's the summary of the activity: summary"
            ),
        ),
        (
            False,
            "A multimodal AI assistant is helping users with some activities. Below is their conversation, interleaved with the list of video frames received by the assistant.",
        ),
    ],
)
def test_construct_system_message(use_narration: bool, expected: str) -> None:
    sys_msg = _construct_system_message(
        [
            {"role": "system", "eval": False, "content": "summary", "start": 0},
            {"role": "assistant", "eval": False, "start": 3},
            {"role": "assistant", "eval": False, "start": 5},
            {"role": "assistant", "eval": True, "start": 8},
        ],
        use_narration,
    )
    assert sys_msg == expected
