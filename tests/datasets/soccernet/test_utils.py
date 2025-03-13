import pytest

from real_time_vlm_benchmark.datasets.soccernet.utils import convert_pbp_annotated


@pytest.mark.parametrize(
    "pbp_annotated,expected",
    [
        (
            {
                "segments": [
                    {"start": 1, "end": 2, "text": "commentary 0", "is_pbp": True},
                    {"start": 2, "end": 4, "text": "commentary 1", "is_pbp": False},
                ]
            },
            [
                {
                    "role": "assistant",
                    "content": "commentary 0",
                    "start": 1,
                    "end": 2,
                    "eval": True,
                },
                {
                    "role": "assistant",
                    "content": "commentary 1",
                    "start": 2,
                    "end": 4,
                    "eval": True,
                },
            ],
        ),
        (
            {
                "segments": [
                    {"start": 1, "end": 2, "text": "commentary 0", "is_pbp": True},
                    {"start": 2, "end": 4, "text": "commentary 1", "is_pbp": False},
                    {"start": 10, "end": 14, "text": "commentary 2", "is_pbp": False},
                    {"start": 15, "end": 17, "text": "commentary 3", "is_pbp": False},
                    {"start": 18, "end": 19, "text": "commentary 4", "is_pbp": True},
                    {"start": 19, "end": 20, "text": "commentary 5", "is_pbp": True},
                    {"start": 20, "end": 22, "text": "commentary 6", "is_pbp": False},
                    {"start": 23, "end": 25, "text": "commentary 7", "is_pbp": True},
                    {"start": 25, "end": 26, "text": "commentary 8", "is_pbp": True},
                    {"start": 30, "end": 32, "text": "commentary 9", "is_pbp": False},
                    {"start": 32, "end": 34, "text": "commentary 10", "is_pbp": False},
                ]
            },
            [
                {
                    "role": "assistant",
                    "content": "commentary 0",
                    "start": 1,
                    "end": 2,
                    "eval": True,
                },
                {
                    "role": "assistant",
                    "content": "commentary 1",
                    "start": 2,
                    "end": 4,
                    "eval": True,
                },
                {
                    "role": "assistant",
                    "content": "commentary 2",
                    "start": 10,
                    "end": 14,
                    "eval": True,
                },
                {
                    "role": "assistant",
                    "content": "commentary 3",
                    "start": 15,
                    "end": 17,
                    "eval": True,
                },
                {
                    "role": "assistant",
                    "content": "commentary 4",
                    "start": 18,
                    "end": 19,
                    "eval": True,
                },
                {
                    "role": "assistant",
                    "content": "commentary 5",
                    "start": 19,
                    "end": 20,
                    "eval": True,
                },
                {
                    "role": "assistant",
                    "content": "commentary 6",
                    "start": 20,
                    "end": 22,
                    "eval": True,
                },
                {
                    "role": "assistant",
                    "content": "commentary 7",
                    "start": 23,
                    "end": 25,
                    "eval": True,
                },
                {
                    "role": "assistant",
                    "content": "commentary 8",
                    "start": 25,
                    "end": 26,
                    "eval": True,
                },
                {
                    "role": "assistant",
                    "content": "commentary 9",
                    "start": 30,
                    "end": 32,
                    "eval": True,
                },
            ],
        ),
    ],
)
def test_convert_pbp_annotated(pbp_annotated: dict, expected: list[dict]) -> None:
    assert convert_pbp_annotated(pbp_annotated) == expected
