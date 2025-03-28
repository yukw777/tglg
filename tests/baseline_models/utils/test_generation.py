import pytest
import torch
from transformers import AutoTokenizer

from real_time_vlm_benchmark.baseline_models.utils.generation import (
    construct_interleaved_dialogue,
    tokenize_real_time_interleaved_dialogue,
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


@pytest.mark.parametrize(
    "interleaved_dialogue,num_total_frames,num_interleaved_frames,train,expected_tokens,expected_num_interleaved_frames",
    [
        (
            [
                {"role": "system", "content": "system message"},
                {"role": "stream", "num_frames": 7},
                {"role": "user", "content": "user utterance 0", "start": 3, "end": 4},
                {"role": "stream", "num_frames": 6},
                {
                    "role": "assistant",
                    "content": "assistant utterance 0",
                    "start": 6.2,
                    "end": 7.5,
                },
            ],
            18,
            13,
            False,
            ["<|begin_of_text|>", "system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3 * 7
            + ["Ċ", "User", ":"]
            + ["<v>"] * 3
            + ["Ġuser", "Ġutter"]
            + ["<v>"] * 3
            + ["ance", "Ġ", "0"]
            + ["<v>"] * 3 * 4
            + ["Ċ", "Assistant", ":"]
            + ["<v>"] * 3
            + ["Ġassistant"]
            + ["Ġutter"]
            + ["<v>"] * 3
            + ["ance", "Ġ"]
            + ["<v>"] * 3
            + ["0", "<|eot_id|>"],
            16,
        ),
        (
            [
                {"role": "system", "content": "system message"},
                {"role": "stream", "num_frames": 1},
                {"role": "user", "content": "user utterance 0", "start": 3, "end": 4},
                {"role": "stream", "num_frames": 6},
                {
                    "role": "assistant",
                    "content": "assistant utterance 0",
                    "start": 6.2,
                    "end": 18.6,
                },
            ],
            25,
            7,
            False,
            ["<|begin_of_text|>", "system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3
            + ["Ċ", "User", ":"]
            + ["<v>"] * 3
            + ["Ġuser", "Ġutter"]
            + ["<v>"] * 3
            + ["ance", "Ġ", "0"]
            + ["<v>"] * 3 * 4
            + ["Ċ", "Assistant", ":"]
            + ["<v>"] * 3
            + ["Ġassistant", "Ġutter"]
            + ["<v>"] * 3
            + ["ance", "Ġ"]
            + ["<v>"] * 3
            + ["0", "<|eot_id|>"],
            10,
        ),
        (
            [
                {"role": "system", "content": "system message"},
                {"role": "stream", "num_frames": 7},
                {"role": "user", "content": "user utterance 0", "start": 3, "end": 4},
                {"role": "stream", "num_frames": 6},
                {
                    "role": "assistant",
                    "content": "assistant utterance 0",
                    "start": 6.2,
                    "end": 7.5,
                },
            ],
            16,
            13,
            False,
            ["<|begin_of_text|>", "system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3 * 7
            + ["Ċ", "User", ":"]
            + ["<v>"] * 3
            + ["Ġuser", "Ġutter"]
            + ["<v>"] * 3
            + ["ance", "Ġ", "0"]
            + ["<v>"] * 3 * 4
            + ["Ċ", "Assistant", ":"]
            + ["<v>"] * 3
            + ["Ġassistant"]
            + ["Ġutter"]
            + ["<v>"] * 3
            + ["ance", "Ġ"]
            + ["<v>"] * 3
            + ["0", "<|eot_id|>"],
            16,
        ),
        (
            [
                {"role": "system", "content": "system message"},
                {"role": "stream", "num_frames": 1},
                {"role": "user", "content": "user utterance 0", "start": 3, "end": 4},
            ],
            1,
            1,
            False,
            ["<|begin_of_text|>", "system", "Ġmessage", "Ċ"] + ["<v>"] * 3,
            1,
        ),
        (
            [
                {"role": "system", "content": "system message"},
                {"role": "stream", "num_frames": 1},
                {"role": "user", "content": "user utterance 0", "start": 3, "end": 4},
            ],
            1,
            1,
            True,
            ["<|begin_of_text|>", "system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3
            + ["<|eot_id|>"],
            1,
        ),
    ],
)
def test_tokenize_real_time_interleaved_dialogue(
    interleaved_dialogue: list[dict],
    num_total_frames: int,
    num_interleaved_frames: int,
    train: bool,
    expected_tokens: list[str],
    expected_num_interleaved_frames: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained("chenjoya/videollm-online-8b-v1plus")
    tokenized, new_num_remaining_frames = tokenize_real_time_interleaved_dialogue(
        tokenizer,
        tokenizer.convert_tokens_to_ids("<v>"),
        3,
        2,
        num_total_frames,
        num_interleaved_frames,
        interleaved_dialogue,
        train=train,
    )
    assert tokenized.tolist() == tokenizer.convert_tokens_to_ids(expected_tokens)
    assert new_num_remaining_frames == expected_num_interleaved_frames
