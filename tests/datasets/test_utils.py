import pytest

from real_time_vlm_benchmark.datasets.utils import convert_real_time_anns_to_datapoint


@pytest.mark.parametrize(
    "anns,expected",
    [
        (
            {
                "video0": [
                    {"role": "system", "content": "system message", "eval": False},
                    {"role": "assistant", "content": "a0", "eval": False},
                    {"role": "assistant", "content": "a1", "eval": False},
                    {"role": "user", "content": "u0", "eval": False},
                    {"role": "assistant", "content": "a2", "eval": True},
                ]
            },
            [
                (
                    "video0",
                    [
                        {"role": "system", "content": "system message", "eval": False},
                        {"role": "assistant", "content": "a0", "eval": False},
                        {"role": "assistant", "content": "a1", "eval": False},
                        {"role": "user", "content": "u0", "eval": False},
                        {"role": "assistant", "content": "a2", "eval": True},
                    ],
                )
            ],
        ),
        (
            {
                "video0": [
                    {"role": "system", "content": "system message", "eval": False},
                    {"role": "assistant", "content": "a0", "eval": False},
                    {"role": "assistant", "content": "a1", "eval": False},
                    {"role": "user", "content": "u0", "eval": False},
                    {"role": "assistant", "content": "a2", "eval": True},
                    {"role": "assistant", "content": "a3", "eval": True},
                    {"role": "assistant", "content": "a4", "eval": True},
                    {"role": "assistant", "content": "a5", "eval": False},
                    {"role": "assistant", "content": "a6", "eval": False},
                    {"role": "assistant", "content": "a7", "eval": True},
                    {"role": "assistant", "content": "a8", "eval": True},
                ],
                "video1": [
                    {"role": "system", "content": "system message", "eval": False},
                    {"role": "assistant", "content": "a0", "eval": False},
                    {"role": "assistant", "content": "a1", "eval": False},
                    {"role": "user", "content": "u0", "eval": False},
                    {"role": "assistant", "content": "a2", "eval": True},
                    {"role": "assistant", "content": "a3", "eval": True},
                    {"role": "assistant", "content": "a4", "eval": False},
                    {"role": "assistant", "content": "a5", "eval": False},
                    {"role": "user", "content": "u1", "eval": False},
                    {"role": "assistant", "content": "a6", "eval": False},
                    {"role": "assistant", "content": "a7", "eval": True},
                    {"role": "assistant", "content": "a8", "eval": True},
                ],
            },
            [
                (
                    "video0",
                    [
                        {"role": "system", "content": "system message", "eval": False},
                        {"role": "assistant", "content": "a0", "eval": False},
                        {"role": "assistant", "content": "a1", "eval": False},
                        {"role": "user", "content": "u0", "eval": False},
                        {"role": "assistant", "content": "a2", "eval": True},
                        {"role": "assistant", "content": "a3", "eval": True},
                        {"role": "assistant", "content": "a4", "eval": True},
                    ],
                ),
                (
                    "video0",
                    [
                        {"role": "system", "content": "system message", "eval": False},
                        {"role": "assistant", "content": "a0", "eval": False},
                        {"role": "assistant", "content": "a1", "eval": False},
                        {"role": "user", "content": "u0", "eval": False},
                        {"role": "assistant", "content": "a2", "eval": False},
                        {"role": "assistant", "content": "a3", "eval": False},
                        {"role": "assistant", "content": "a4", "eval": False},
                        {"role": "assistant", "content": "a5", "eval": False},
                        {"role": "assistant", "content": "a6", "eval": False},
                        {"role": "assistant", "content": "a7", "eval": True},
                        {"role": "assistant", "content": "a8", "eval": True},
                    ],
                ),
                (
                    "video1",
                    [
                        {"role": "system", "content": "system message", "eval": False},
                        {"role": "assistant", "content": "a0", "eval": False},
                        {"role": "assistant", "content": "a1", "eval": False},
                        {"role": "user", "content": "u0", "eval": False},
                        {"role": "assistant", "content": "a2", "eval": True},
                        {"role": "assistant", "content": "a3", "eval": True},
                    ],
                ),
                (
                    "video1",
                    [
                        {"role": "system", "content": "system message", "eval": False},
                        {"role": "assistant", "content": "a0", "eval": False},
                        {"role": "assistant", "content": "a1", "eval": False},
                        {"role": "user", "content": "u0", "eval": False},
                        {"role": "assistant", "content": "a2", "eval": False},
                        {"role": "assistant", "content": "a3", "eval": False},
                        {"role": "assistant", "content": "a4", "eval": False},
                        {"role": "assistant", "content": "a5", "eval": False},
                        {"role": "user", "content": "u1", "eval": False},
                        {"role": "assistant", "content": "a6", "eval": False},
                        {"role": "assistant", "content": "a7", "eval": True},
                        {"role": "assistant", "content": "a8", "eval": True},
                    ],
                ),
            ],
        ),
    ],
)
def test_convert_real_time_anns_to_datapoint(
    anns: dict[str, dict], expected: list[tuple[str, list[dict]]]
) -> None:
    assert convert_real_time_anns_to_datapoint(anns) == expected
