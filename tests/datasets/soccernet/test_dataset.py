import pytest

from real_time_vlm_benchmark.datasets.soccernet.dataset import (
    _convert_real_time_anns_to_datapoint,
)


@pytest.mark.parametrize(
    "anns,expected",
    [
        (
            {
                "video0": [
                    {"role": "assistant", "content": "a0", "start": 0, "end": 2},
                    {"role": "assistant", "content": "a1", "start": 2.5, "end": 4},
                    {"role": "assistant", "content": "a2", "start": 4.2, "end": 6},
                ]
            },
            [
                (
                    "video0",
                    [
                        {"role": "assistant", "content": "a0", "start": 0, "end": 2},
                        {"role": "assistant", "content": "a1", "start": 2.5, "end": 4},
                        {"role": "assistant", "content": "a2", "start": 4.2, "end": 6},
                    ],
                )
            ],
        ),
        (
            {
                "video0": [
                    {"role": "assistant", "content": "a0", "start": 0, "end": 2},
                    {"role": "assistant", "content": "a1", "start": 2.5, "end": 4},
                    {"role": "assistant", "content": "a2", "start": 4.2, "end": 6},
                    {"role": "assistant", "content": "a3", "start": 12, "end": 14},
                    {"role": "assistant", "content": "a4", "start": 14.5, "end": 17},
                    {"role": "assistant", "content": "a5", "start": 18, "end": 20},
                    {"role": "assistant", "content": "a6", "start": 21, "end": 23},
                ],
                "video1": [
                    {"role": "assistant", "content": "a0", "start": 0, "end": 2},
                    {"role": "assistant", "content": "a1", "start": 2.5, "end": 4},
                    {"role": "assistant", "content": "a2", "start": 4.2, "end": 6},
                    {"role": "assistant", "content": "a3", "start": 12, "end": 14},
                    {"role": "assistant", "content": "a4", "start": 14.5, "end": 17},
                    {"role": "assistant", "content": "a5", "start": 18, "end": 20},
                    {"role": "assistant", "content": "a6", "start": 21, "end": 23},
                    {"role": "assistant", "content": "a7", "start": 40, "end": 42},
                    {"role": "assistant", "content": "a8", "start": 43, "end": 50},
                    {"role": "assistant", "content": "a9", "start": 51, "end": 52},
                    {"role": "assistant", "content": "a10", "start": 52.5, "end": 55},
                ],
            },
            [
                (
                    "video0",
                    [
                        {"role": "assistant", "content": "a0", "start": 0, "end": 2},
                        {"role": "assistant", "content": "a1", "start": 2.5, "end": 4},
                        {"role": "assistant", "content": "a2", "start": 4.2, "end": 6},
                    ],
                ),
                (
                    "video0",
                    [
                        {"role": "assistant", "content": "a3", "start": 12, "end": 14},
                        {
                            "role": "assistant",
                            "content": "a4",
                            "start": 14.5,
                            "end": 17,
                        },
                        {"role": "assistant", "content": "a5", "start": 18, "end": 20},
                        {"role": "assistant", "content": "a6", "start": 21, "end": 23},
                    ],
                ),
                (
                    "video1",
                    [
                        {"role": "assistant", "content": "a0", "start": 0, "end": 2},
                        {"role": "assistant", "content": "a1", "start": 2.5, "end": 4},
                        {"role": "assistant", "content": "a2", "start": 4.2, "end": 6},
                    ],
                ),
                (
                    "video1",
                    [
                        {"role": "assistant", "content": "a3", "start": 12, "end": 14},
                        {
                            "role": "assistant",
                            "content": "a4",
                            "start": 14.5,
                            "end": 17,
                        },
                        {"role": "assistant", "content": "a5", "start": 18, "end": 20},
                        {"role": "assistant", "content": "a6", "start": 21, "end": 23},
                    ],
                ),
                (
                    "video1",
                    [
                        {"role": "assistant", "content": "a7", "start": 40, "end": 42},
                        {"role": "assistant", "content": "a8", "start": 43, "end": 50},
                        {"role": "assistant", "content": "a9", "start": 51, "end": 52},
                        {
                            "role": "assistant",
                            "content": "a10",
                            "start": 52.5,
                            "end": 55,
                        },
                    ],
                ),
            ],
        ),
    ],
)
def test_soccernet_convert_real_time_anns_to_datapoint(
    anns: dict[str, list[dict]], expected: list[tuple[str, list[dict]]]
) -> None:
    assert _convert_real_time_anns_to_datapoint(anns) == expected
