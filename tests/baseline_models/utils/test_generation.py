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
    "interleaved_dialogue,num_total_frames,num_interleaved_frames,expected_tokens,expected_labels,expected_num_interleaved_frames",
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
                {"role": "stream", "num_frames": 2},
            ],
            18,
            15,
            ["system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3 * 7
            + ["Ċ", "User", ":", "Ġuser", "Ġutter", "ance"]
            + ["<v>"] * 3
            + ["Ġ", "0"]
            + ["<v>"] * 3 * 5
            + ["Ċ", "Assistant", ":", "Ġassistant", "Ġutter", "ance"]
            + ["<v>"] * 3
            + ["Ġ", "0"],
            [-100, -100, -100]
            + [-100] * 3 * 7
            + [-100, -100, -100, -100, -100, -100]
            + [-100] * 3
            + [-100, -100]
            + [-100] * 3 * 4
            + [-100] * 2
            + ["Ċ", "Assistant", ":", "Ġassistant", "Ġutter", "ance"]
            + [-100] * 3
            + ["Ġ", "0", "<|eot_id|>"],
            14,
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
            18,
            13,
            ["system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3 * 7
            + ["Ċ", "User", ":", "Ġuser", "Ġutter", "ance"]
            + ["<v>"] * 3
            + ["Ġ", "0"]
            + ["<v>"] * 3 * 5
            + ["Ċ", "Assistant", ":", "Ġassistant", "Ġutter"]
            + ["<v>"] * 3
            + ["ance", "Ġ"]
            + ["<v>"] * 3
            + ["0"],
            [-100, -100, -100]
            + [-100] * 3 * 7
            + [-100, -100, -100, -100, -100, -100]
            + [-100] * 3
            + [-100, -100]
            + [-100] * 3 * 4
            + [-100] * 2
            + ["Ċ", "Assistant", ":", "Ġassistant", "Ġutter"]
            + [-100] * 3
            + ["ance", "Ġ"]
            + [-100] * 3
            + ["0", "<|eot_id|>"],
            15,
        ),
        (
            [
                {"role": "system", "content": "system message"},
                {"role": "stream", "num_frames": 7},
                {"role": "user", "content": "utter", "start": 3, "end": 4},
                {"role": "stream", "num_frames": 6},
                {
                    "role": "assistant",
                    "content": "utter",
                    "start": 6.2,
                    "end": 7.5,
                },
            ],
            18,
            13,
            ["system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3 * 7
            + ["Ċ", "User", ":", "Ġutter"]
            + ["<v>"] * 3 * 6
            + ["Ċ", "Assistant", ":", "Ġutter"],
            [-100, -100, -100]
            + [-100] * 3 * 7
            + [-100, -100, -100, -100]
            + [-100] * 3 * 5
            + [-100] * 2
            + ["Ċ", "Assistant", ":", "Ġutter", "<|eot_id|>"],
            13,
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
            ["system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3
            + ["Ċ", "User", ":", "Ġuser", "Ġutter", "ance"]
            + ["<v>"] * 3
            + ["Ġ", "0"]
            + ["<v>"] * 3 * 5
            + ["Ċ", "Assistant", ":", "Ġassistant", "Ġutter"]
            + ["<v>"] * 3
            + ["ance", "Ġ"]
            + ["<v>"] * 3
            + ["0"],
            [-100, -100, -100]
            + [-100] * 3
            + [-100, -100, -100, -100, -100, -100]
            + [-100] * 3
            + [-100, -100]
            + [-100] * 3 * 4
            + [-100] * 2
            + ["Ċ", "Assistant", ":", "Ġassistant", "Ġutter"]
            + [-100] * 3
            + ["ance", "Ġ"]
            + [-100] * 3
            + ["0", "<|eot_id|>"],
            9,
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
            ["system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3 * 7
            + ["Ċ", "User", ":", "Ġuser", "Ġutter", "ance"]
            + ["<v>"] * 3
            + ["Ġ", "0"]
            + ["<v>"] * 3 * 5
            + ["Ċ", "Assistant", ":", "Ġassistant", "Ġutter"]
            + ["<v>"] * 3
            + ["ance", "Ġ"]
            + ["<v>"] * 3
            + ["0"],
            [-100, -100, -100]
            + [-100] * 3 * 7
            + [-100, -100, -100, -100, -100]
            + [-100] * 3
            + [-100, -100]
            + [-100] * 3
            + [-100]
            + [-100] * 3 * 3
            + [-100, -100]
            + ["Ċ", "Assistant", ":", "Ġassistant", "Ġutter"]
            + [-100] * 3
            + ["ance", "Ġ"]
            + [-100] * 3
            + ["0", "<|eot_id|>"],
            15,
        ),
        (
            [
                {"role": "system", "content": "system message"},
                {"role": "stream", "num_frames": 1},
                {"role": "user", "content": "user utterance 0", "start": 3, "end": 4},
            ],
            1,
            1,
            ["system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3
            + ["Ċ", "User", ":", "Ġuser", "Ġutter", "ance", "Ġ", "0"],
            [-100, -100, -100]
            + [-100] * 3
            + [-100, -100, -100, -100, -100, -100, -100, -100],
            1,
        ),
        (
            [
                {"role": "system", "content": "system message"},
                {"role": "stream", "num_frames": 7},
                {"role": "user", "content": "utter", "start": 3, "end": 3.5},
                {"role": "stream", "num_frames": 6},
                {
                    "role": "assistant",
                    "content": "assistant utterance 0",
                    "start": 6.2,
                    "end": 7.3,
                },
                {"role": "stream", "num_frames": 5},
                {
                    "role": "user",
                    "content": "user utterance 0",
                    "start": 8.5,
                    "end": 9.5,
                },
                {"role": "stream", "num_frames": 4},
                {
                    "role": "assistant",
                    "content": "utter",
                    "start": 11.2,
                    "end": 12,
                },
            ],
            30,
            22,
            ["system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3 * 7
            + ["Ċ", "User", ":", "Ġutter"]
            + ["<v>"] * 3 * 6
            + ["Ċ", "Assistant", ":", "Ġassistant", "Ġutter"]
            + ["<v>"] * 3
            + ["ance", "Ġ"]
            + ["<v>"] * 3
            + ["0"]
            + ["<v>"] * 3 * 3
            + ["Ċ", "User", ":", "Ġuser", "Ġutter", "ance"]
            + ["<v>"] * 3
            + ["Ġ", "0"]
            + ["<v>"] * 3 * 3
            + ["Ċ", "Assistant", ":", "Ġutter"],
            [-100, -100, -100]
            + [-100] * 3 * 7
            + [-100, -100, -100, -100]
            + [-100] * 3 * 5
            + [-100] * 2
            + ["Ċ", "Assistant", ":", "Ġassistant", "Ġutter"]
            + [-100] * 3
            + ["ance", "Ġ"]
            + [-100] * 3
            + ["0", "<|eot_id|>"]
            + [-100] * 3 * 3
            + [-100, -100, -100, -100, -100, -100]
            + [-100] * 3
            + [-100, -100]
            + [-100] * 3 * 2
            + [-100, -100]
            + ["Ċ", "Assistant", ":", "Ġutter", "<|eot_id|>"],
            22,
        ),
        (
            [
                {"role": "system", "content": "system message"},
                {"role": "stream", "num_frames": 1},
                {
                    "role": "assistant",
                    "content": "Now rolls the reverse.",
                    "start": 1789.267,
                    "end": 1790.227,
                    "eval": True,
                },
                {"role": "stream", "num_frames": 1},
                {
                    "role": "assistant",
                    "content": "Mascherano is clearly still feeling it.",
                    "start": 1790.288,
                    "end": 1792.129,
                    "eval": True,
                },
            ],
            6,
            2,
            ["system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3
            + ["Ċ", "Assistant", ":", "ĠNow", "Ġrolls", "Ġthe", "Ġreverse", "."]
            + ["<v>"] * 3
            + ["Ċ", "Assistant", ":", "ĠMas", "cher", "ano"]
            + ["<v>"] * 3
            + ["Ġis", "Ġclearly", "Ġstill"]
            + ["<v>"] * 3
            + ["Ġfeeling", "Ġit", "."],
            [-100, -100, -100]
            + [-100] * 2
            + [
                "Ċ",
                "Assistant",
                ":",
                "ĠNow",
                "Ġrolls",
                "Ġthe",
                "Ġreverse",
                ".",
                "<|eot_id|>",
            ]
            + [-100] * 2
            + ["Ċ", "Assistant", ":", "ĠMas", "cher", "ano"]
            + [-100] * 3
            + ["Ġis", "Ġclearly", "Ġstill"]
            + [-100] * 3
            + ["Ġfeeling", "Ġit", ".", "<|eot_id|>"],
            4,
        ),
        (
            [
                {"role": "system", "content": "system message"},
                {"role": "stream", "num_frames": 1},
                {
                    "role": "assistant",
                    "content": " Hard to deny to Messi for the penalty spot and stop it's time to keep the tie alive in the first leg.",
                    "start": 913.092,
                    "end": 918.056,
                    "eval": True,
                },
                {"role": "stream", "num_frames": 9},
                {
                    "role": "assistant",
                    "content": "It's Messi.",
                    "start": 918.116,
                    "end": 920.178,
                    "eval": True,
                },
                {"role": "stream", "num_frames": 7},
                {
                    "role": "assistant",
                    "content": "Ooh, sail just over the top.",
                    "start": 921.179,
                    "end": 922.58,
                    "eval": True,
                },
                {"role": "stream", "num_frames": 3},
                {
                    "role": "assistant",
                    "content": "It wasn't far away, was it?",
                    "start": 922.74,
                    "end": 925.042,
                    "eval": True,
                },
                {"role": "stream", "num_frames": 4},
                {
                    "role": "assistant",
                    "content": "He was plead with it to come down a little bit quicker, to catch us at the top of the net.",
                    "start": 925.062,
                    "end": 931.307,
                    "eval": True,
                },
            ],
            37,
            24,
            ["system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3
            + ["Ċ", "Assistant", ":", "Ġ", "ĠHard", "Ġto"]
            + ["<v>"] * 3
            + ["Ġdeny", "Ġto", "ĠMessi"]
            + ["<v>"] * 3
            + ["Ġfor", "Ġthe", "Ġpenalty"]
            + ["<v>"] * 3
            + ["Ġspot", "Ġand", "Ġstop"]
            + ["<v>"] * 3
            + ["Ġit", "'s", "Ġtime"]
            + ["<v>"] * 3
            + ["Ġto", "Ġkeep", "Ġthe"]
            + ["<v>"] * 3
            + ["Ġtie", "Ġalive", "Ġin"]
            + ["<v>"] * 3
            + ["Ġthe", "Ġfirst", "Ġleg"]
            + ["<v>"] * 3
            + ["."]
            + ["<v>"] * 3
            + ["Ċ", "Assistant", ":", "ĠIt"]
            + ["<v>"] * 3
            + ["'s"]
            + ["<v>"] * 3
            + ["ĠMessi"]
            + ["<v>"] * 3
            + ["."]
            + ["<v>"] * 3 * 4
            + ["Ċ", "Assistant", ":", "ĠO", "oh", ","]
            + ["<v>"] * 3
            + ["Ġsail", "Ġjust", "Ġover"]
            + ["<v>"] * 3
            + ["Ġthe", "Ġtop", "."]
            + ["<v>"] * 3
            + ["Ċ", "Assistant", ":", "ĠIt", "Ġwasn", "'t"]
            + ["<v>"] * 3
            + ["Ġfar", "Ġaway", ","]
            + ["<v>"] * 3
            + ["Ġwas", "Ġit", "?"]
            + ["<v>"] * 3 * 2
            + ["Ċ", "Assistant", ":", "ĠHe", "Ġwas"]
            + ["<v>"] * 3
            + ["Ġplead", "Ġwith"]
            + ["<v>"] * 3
            + ["Ġit", "Ġto"]
            + ["<v>"] * 3
            + ["Ġcome", "Ġdown"]
            + ["<v>"] * 3
            + ["Ġa", "Ġlittle"]
            + ["<v>"] * 3
            + ["Ġbit", "Ġquicker"]
            + ["<v>"] * 3
            + [",", "Ġto"]
            + ["<v>"] * 3
            + ["Ġcatch", "Ġus"]
            + ["<v>"] * 3
            + ["Ġat", "Ġthe"]
            + ["<v>"] * 3
            + ["Ġtop", "Ġof"]
            + ["<v>"] * 3
            + ["Ġthe", "Ġnet"]
            + ["<v>"] * 3
            + ["."],
            [-100, -100, -100]
            + [-100] * 2
            + ["Ċ", "Assistant", ":", "Ġ", "ĠHard", "Ġto"]
            + [-100] * 3
            + ["Ġdeny", "Ġto", "ĠMessi"]
            + [-100] * 3
            + ["Ġfor", "Ġthe", "Ġpenalty"]
            + [-100] * 3
            + ["Ġspot", "Ġand", "Ġstop"]
            + [-100] * 3
            + ["Ġit", "'s", "Ġtime"]
            + [-100] * 3
            + ["Ġto", "Ġkeep", "Ġthe"]
            + [-100] * 3
            + ["Ġtie", "Ġalive", "Ġin"]
            + [-100] * 3
            + ["Ġthe", "Ġfirst", "Ġleg"]
            + [-100] * 3
            + [".", "<|eot_id|>"]
            + [-100] * 2
            + ["Ċ", "Assistant", ":", "ĠIt"]
            + [-100] * 3
            + ["'s"]
            + [-100] * 3
            + ["ĠMessi"]
            + [-100] * 3
            + [".", "<|eot_id|>"]
            + [-100] * 3 * 3
            + [-100] * 2
            + ["Ċ", "Assistant", ":", "ĠO", "oh", ","]
            + [-100] * 3
            + ["Ġsail", "Ġjust", "Ġover"]
            + [-100] * 3
            + ["Ġthe", "Ġtop", ".", "<|eot_id|>"]
            + [-100] * 2
            + ["Ċ", "Assistant", ":", "ĠIt", "Ġwasn", "'t"]
            + [-100] * 3
            + ["Ġfar", "Ġaway", ","]
            + [-100] * 3
            + ["Ġwas", "Ġit", "?", "<|eot_id|>"]
            + [-100] * 3
            + [-100] * 2
            + ["Ċ", "Assistant", ":", "ĠHe", "Ġwas"]
            + [-100] * 3
            + ["Ġplead", "Ġwith"]
            + [-100] * 3
            + ["Ġit", "Ġto"]
            + [-100] * 3
            + ["Ġcome", "Ġdown"]
            + [-100] * 3
            + ["Ġa", "Ġlittle"]
            + [-100] * 3
            + ["Ġbit", "Ġquicker"]
            + [-100] * 3
            + [",", "Ġto"]
            + [-100] * 3
            + ["Ġcatch", "Ġus"]
            + [-100] * 3
            + ["Ġat", "Ġthe"]
            + [-100] * 3
            + ["Ġtop", "Ġof"]
            + [-100] * 3
            + ["Ġthe", "Ġnet"]
            + [-100] * 3
            + [".", "<|eot_id|>"],
            35,
        ),
        (
            [
                {"role": "system", "content": "system message"},
                {"role": "stream", "num_frames": 1},
                {
                    "role": "assistant",
                    "content": "Monreal.",
                    "start": 325.377,
                    "end": 325.657,
                    "eval": True,
                },
            ],
            1,
            1,
            ["system", "Ġmessage", "Ċ"]
            + ["<v>"] * 3
            + ["Ċ", "Assistant", ":", "ĠMon", "real", "."],
            [-100, -100, -100]
            + [-100] * 2
            + ["Ċ", "Assistant", ":", "ĠMon", "real", ".", "<|eot_id|>"],
            1,
        ),
    ],
)
def test_tokenize_real_time_interleaved_dialogue(
    interleaved_dialogue: list[dict],
    num_total_frames: int,
    num_interleaved_frames: int,
    expected_tokens: list[str],
    expected_labels: list[str | int],
    expected_num_interleaved_frames: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained("chenjoya/videollm-online-8b-v1plus")
    tokenized, labels, new_num_interleaved_frames = (
        tokenize_real_time_interleaved_dialogue(
            tokenizer,
            tokenizer.convert_tokens_to_ids("<v>"),
            tokenizer.eos_token_id,
            3,
            2,
            num_total_frames,
            num_interleaved_frames,
            interleaved_dialogue,
        )
    )
    tokenized_list = tokenized.tolist()
    labels_list = labels.tolist()
    assert len(tokenized_list) == len(labels_list)
    assert tokenized_list == tokenizer.convert_tokens_to_ids(expected_tokens)
    assert labels_list == [
        label if isinstance(label, int) else tokenizer.convert_tokens_to_ids(label)
        for label in expected_labels
    ]
    assert new_num_interleaved_frames == expected_num_interleaved_frames
