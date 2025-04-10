import json
from pathlib import Path
from typing import Any

import pytest

from real_time_vlm_benchmark.datasets.holo_assist import (
    HoloAssistDataset,
    _convert_real_time_anns_to_datapoint,
    convert_holo_assist,
)


@pytest.mark.parametrize(
    "holo_assist_anns,expected",
    [
        (
            [
                {
                    "video_name": "video0",
                    "events": [
                        {
                            "label": "Narration",
                            "attributes": {"Long form description": "description"},
                        },
                        {
                            "id": 116,
                            "label": "Conversation",
                            "start": 7.451,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "instructor-start-conversation_other",
                                "sequence": "none",
                                "Transcription": "*unintelligible*",
                                "Transcription Confidence": "low-confidence-transcription",
                                "Conversation Purpose -- Other": "none",
                            },
                            "end": 8.872,
                        },
                        {
                            "id": 117,
                            "label": "Conversation",
                            "start": 9.392,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "instructor-start-conversation_other",
                                "sequence": "none",
                                "Transcription": "Huh?",
                                "Transcription Confidence": "none",
                                "Conversation Purpose -- Other": "none",
                            },
                            "end": 9.902,
                        },
                        {
                            "id": 3,
                            "label": "Coarse grained action",
                            "start": 10.975,
                            "type": "range",
                            "attributes": {
                                "Action sentence": "The student grabs the GoPro.",
                                "Verb": "grab",
                                "Adjective": "none",
                                "Noun": "gopro",
                            },
                            "end": 28.21,
                        },
                    ],
                },
            ],
            {},
        ),
        (
            [
                {
                    "video_name": "video1",
                    "events": [
                        {
                            "label": "Narration",
                            "attributes": {"Long form description": "description"},
                        },
                        {
                            "id": 116,
                            "label": "Conversation",
                            "start": 7.451,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "instructor-start-conversation_other",
                                "sequence": "none",
                                "Transcription": "*unintelligible*",
                                "Transcription Confidence": "low-confidence-transcription",
                                "Conversation Purpose -- Other": "none",
                            },
                            "end": 8.872,
                        },
                        {
                            "id": 117,
                            "label": "Conversation",
                            "start": 9.392,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "instructor-start-conversation_other",
                                "sequence": "none",
                                "Transcription": "Huh?",
                                "Transcription Confidence": "none",
                                "Conversation Purpose -- Other": "none",
                            },
                            "end": 9.902,
                        },
                        {
                            "id": 5,
                            "label": "Conversation",
                            "start": 14.2,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "instructor-start-conversation_describing high-level instruction",
                                "sequence": "none",
                                "Transcription": "You can pull the GoPro.",
                                "Transcription Confidence": "none",
                            },
                            "end": 15.967,
                        },
                        {
                            "id": 120,
                            "label": "Conversation",
                            "start": 16.598,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "student-start-conversation_other",
                                "sequence": "none",
                                "Transcription": "GoPro.",
                                "Transcription Confidence": "none",
                                "Conversation Purpose -- Other": "none",
                            },
                            "end": 17.167,
                        },
                        {
                            "id": 3,
                            "label": "Coarse grained action",
                            "start": 10.975,
                            "type": "range",
                            "attributes": {
                                "Action sentence": "The student grabs the GoPro.",
                                "Verb": "grab",
                                "Adjective": "none",
                                "Noun": "gopro",
                            },
                            "end": 28.21,
                        },
                        {
                            "id": 20,
                            "label": "Conversation",
                            "start": 65.749,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "instructor-start-conversation_describing high-level instruction",
                                "sequence": "none",
                                "Transcription": "Now change the micro SD.",
                                "Transcription Confidence": "none",
                            },
                            "end": 68.343,
                        },
                        {
                            "id": 21,
                            "label": "Coarse grained action",
                            "start": 68.379,
                            "type": "range",
                            "attributes": {
                                "Action sentence": "The student opens the GoPro.",
                                "Verb": "exchange",
                                "Adjective": "none",
                                "Noun": "sd_card",
                            },
                            "end": 304.325,
                        },
                        {
                            "id": 22,
                            "label": "Fine grained action",
                            "start": 68.583,
                            "type": "range",
                            "attributes": {
                                "Action Correctness": "Correct Action",
                                "Incorrect Action Explanation": "none",
                                "Incorrect Action Corrected by": "none",
                                "Verb": "grab",
                                "Adjective": "none",
                                "Noun": "battery_door",
                                "adverbial": "none",
                            },
                            "end": 70.802,
                        },
                        {
                            "id": 132,
                            "label": "Fine grained action",
                            "start": 70.824,
                            "type": "range",
                            "attributes": {
                                "Action Correctness": "Correct Action",
                                "Incorrect Action Explanation": "none",
                                "Incorrect Action Corrected by": "none",
                                "Verb": "open",
                                "Adjective": "none",
                                "Noun": "battery_door",
                                "adverbial": "none",
                            },
                            "end": 72.656,
                        },
                        {
                            "id": 133,
                            "label": "Fine grained action",
                            "start": 72.676,
                            "type": "range",
                            "attributes": {
                                "Action Correctness": "Wrong Action, corrected by instructor verbally",
                                "Incorrect Action Explanation": "The student presses the wrong place.",
                                "Incorrect Action Corrected by": "23",
                                "Verb": "press",
                                "Adjective": "none",
                                "Noun": "battery",
                                "adverbial": "none",
                            },
                            "end": 73.853,
                        },
                        {
                            "id": 23,
                            "label": "Conversation",
                            "start": 73.1,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "instructor-start-conversation_correct the wrong action",
                                "sequence": "133",
                                "Transcription": "No that one.",
                                "Transcription Confidence": "none",
                            },
                            "end": 74.8,
                        },
                    ],
                },
            ],
            {
                "video1": [
                    {"role": "system", "content": "description", "eval": False},
                    {
                        "role": "assistant",
                        "content": "Huh?",
                        "start": 9.392,
                        "end": 9.902,
                        "eval": False,
                    },
                    {
                        "role": "assistant",
                        "content": "You can pull the GoPro.",
                        "start": 14.2,
                        "end": 15.967,
                        "eval": False,
                    },
                    {
                        "role": "user",
                        "content": "GoPro.",
                        "start": 16.598,
                        "end": 17.167,
                        "eval": False,
                    },
                    {
                        "role": "assistant",
                        "start": 65.749,
                        "content": "Now change the micro SD.",
                        "end": 68.343,
                        "eval": True,
                    },
                    {
                        "role": "assistant",
                        "start": 73.1,
                        "content": "No that one.",
                        "end": 74.8,
                        "eval": True,
                    },
                ]
            },
        ),
        (
            [
                {
                    "video_name": "video0",
                    "events": [
                        {
                            "label": "Narration",
                            "attributes": {"Long form description": "description"},
                        },
                        {
                            "id": 58,
                            "label": "Conversation",
                            "start": 198.482,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "instructor-start-conversation_follow-up instruction",
                                "sequence": "none",
                                "Transcription": "Put down the flap.",
                                "Transcription Confidence": "none",
                            },
                            "end": 199.535,
                        },
                        {
                            "id": 59,
                            "label": "Conversation",
                            "start": 200.524,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "student-start-conversation_ask questions",
                                "sequence": "none",
                                "Transcription": "Flip down the flap? This flap?",
                                "Transcription Confidence": "none",
                                "Conversation Purpose -- Other": "none",
                            },
                            "end": 203.359,
                        },
                        {
                            "id": 60,
                            "label": "Fine grained action",
                            "start": 202.206,
                            "type": "range",
                            "attributes": {
                                "Action Correctness": "Wrong Action, corrected by instructor verbally",
                                "Incorrect Action Explanation": "Wrong flap.",
                                "Incorrect Action Corrected by": "63",
                                "Verb": "open",
                                "Adjective": "none",
                                "Noun": "copy_gate",
                                "adverbial": "none",
                            },
                            "end": 203.74,
                        },
                        {
                            "id": 63,
                            "label": "Conversation",
                            "start": 203.482,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "instructor-start-conversation_correct the wrong action",
                                "sequence": "60",
                                "Transcription": "No.",
                                "Transcription Confidence": "none",
                            },
                            "end": 203.947,
                        },
                        {
                            "id": 117,
                            "label": "Fine grained action",
                            "start": 203.871,
                            "type": "range",
                            "attributes": {
                                "Action Correctness": "Correct Action",
                                "Incorrect Action Explanation": "none",
                                "Incorrect Action Corrected by": "none",
                                "Verb": "grab",
                                "Adjective": "none",
                                "Noun": "printer",
                                "adverbial": "none",
                            },
                            "end": 204.9,
                        },
                        {
                            "id": 102,
                            "label": "Conversation",
                            "start": 204.074,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "student-start-conversation_other",
                                "sequence": "none",
                                "Transcription": "No.",
                                "Transcription Confidence": "none",
                                "Conversation Purpose -- Other": "none",
                            },
                            "end": 204.485,
                        },
                        {
                            "id": 64,
                            "label": "Conversation",
                            "start": 205.718,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "student-start-conversation_other",
                                "sequence": "none",
                                "Transcription": "Oh, this flap.",
                                "Transcription Confidence": "none",
                                "Conversation Purpose -- Other": "none",
                            },
                            "end": 207.488,
                        },
                        {
                            "id": 68,
                            "label": "Fine grained action",
                            "start": 207.43,
                            "type": "range",
                            "attributes": {
                                "Action Correctness": "Correct Action",
                                "Incorrect Action Explanation": "none",
                                "Incorrect Action Corrected by": "none",
                                "Verb": "close",
                                "Adjective": "extending",
                                "Noun": "paper_tray",
                                "adverbial": "none",
                            },
                            "end": 208.4,
                        },
                        {
                            "id": 70,
                            "label": "Conversation",
                            "start": 208.835,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "instructor-start-conversation_follow-up instruction",
                                "sequence": "none",
                                "Transcription": "Unfold this paper.",
                                "Transcription Confidence": "none",
                            },
                            "end": 210.588,
                        },
                        {
                            "id": 103,
                            "label": "Conversation",
                            "start": 210.882,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "student-start-conversation_other",
                                "sequence": "none",
                                "Transcription": "Alright.",
                                "Transcription Confidence": "none",
                                "Conversation Purpose -- Other": "Affirmation",
                            },
                            "end": 211.736,
                        },
                        {
                            "id": 71,
                            "label": "Fine grained action",
                            "start": 210.992,
                            "type": "range",
                            "attributes": {
                                "Action Correctness": "Correct Action",
                                "Incorrect Action Explanation": "none",
                                "Incorrect Action Corrected by": "none",
                                "Verb": "grab",
                                "Adjective": "none",
                                "Noun": "copy",
                                "adverbial": "none",
                            },
                            "end": 211.372,
                        },
                        {
                            "id": 73,
                            "label": "Fine grained action",
                            "start": 211.423,
                            "type": "range",
                            "attributes": {
                                "Action Correctness": "Correct Action",
                                "Incorrect Action Explanation": "none",
                                "Incorrect Action Corrected by": "none",
                                "Verb": "lift",
                                "Adjective": "none",
                                "Noun": "copy",
                                "adverbial": "none",
                            },
                            "end": 214.247,
                        },
                        {
                            "id": 74,
                            "label": "Conversation",
                            "start": 211.953,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "instructor-start-conversation_follow-up instruction",
                                "sequence": "none",
                                "Transcription": "Set it nice on the table.",
                                "Transcription Confidence": "none",
                            },
                            "end": 214.335,
                        },
                        {
                            "id": 76,
                            "label": "Fine grained action",
                            "start": 214.307,
                            "type": "range",
                            "attributes": {
                                "Action Correctness": "Correct Action",
                                "Incorrect Action Explanation": "none",
                                "Incorrect Action Corrected by": "none",
                                "Verb": "place",
                                "Adjective": "none",
                                "Noun": "paper_stack",
                                "adverbial": "none",
                            },
                            "end": 214.661,
                        },
                        {
                            "id": 75,
                            "label": "Conversation",
                            "start": 214.512,
                            "type": "range",
                            "attributes": {
                                "Conversation Purpose": "student-start-conversation_other",
                                "sequence": "none",
                                "Transcription": "Okay.",
                                "Transcription Confidence": "none",
                                "Conversation Purpose -- Other": "Affirmation",
                            },
                            "end": 215.212,
                        },
                        {
                            "id": 77,
                            "label": "Fine grained action",
                            "start": 214.708,
                            "type": "range",
                            "attributes": {
                                "Action Correctness": "Correct Action",
                                "Incorrect Action Explanation": "none",
                                "Incorrect Action Corrected by": "none",
                                "Verb": "align",
                                "Adjective": "none",
                                "Noun": "copy",
                                "adverbial": "none",
                            },
                            "end": 215.8,
                        },
                    ],
                }
            ],
            {
                "video0": [
                    {"role": "system", "content": "description", "eval": False},
                    {
                        "role": "assistant",
                        "content": "Put down the flap.",
                        "start": 198.482,
                        "end": 199.535,
                        "eval": True,
                    },
                    {
                        "role": "user",
                        "content": "Flip down the flap? This flap?",
                        "start": 200.524,
                        "end": 203.359,
                        "eval": False,
                    },
                    {
                        "role": "assistant",
                        "content": "No.",
                        "start": 203.482,
                        "end": 203.947,
                        "eval": True,
                    },
                    {
                        "role": "user",
                        "content": "No.",
                        "start": 204.074,
                        "end": 204.485,
                        "eval": False,
                    },
                    {
                        "role": "user",
                        "content": "Oh, this flap.",
                        "start": 205.718,
                        "end": 207.488,
                        "eval": False,
                    },
                    {
                        "role": "assistant",
                        "content": "Unfold this paper.",
                        "start": 208.835,
                        "end": 210.588,
                        "eval": True,
                    },
                    {
                        "role": "user",
                        "content": "Alright.",
                        "start": 210.882,
                        "end": 211.736,
                        "eval": False,
                    },
                    {
                        "role": "assistant",
                        "content": "Set it nice on the table.",
                        "start": 211.953,
                        "end": 214.335,
                        "eval": True,
                    },
                ]
            },
        ),
    ],
)
def test_convert_holo_assist(
    holo_assist_anns: list[dict], expected: dict[str, list[dict[str, Any]]]
) -> None:
    assert convert_holo_assist(holo_assist_anns) == expected


@pytest.mark.parametrize(
    "original_anns,video_frame_dir_path,expected_items",
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
            None,
            [
                {
                    "index": 0,
                    "video_id": "video0",
                    "video_path": Path(
                        "video_dir/video0/Export_py/Video_pitchshift.mp4"
                    ),
                    "dialogue": [
                        {"role": "system", "content": "system message", "eval": False},
                        {"role": "assistant", "content": "a0", "eval": False},
                        {"role": "assistant", "content": "a1", "eval": False},
                        {"role": "user", "content": "u0", "eval": False},
                        {"role": "assistant", "content": "a2", "eval": True},
                    ],
                }
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
                ]
            },
            Path("video_frame_dir_path"),
            [
                {
                    "index": 0,
                    "video_id": "video0",
                    "video_path": Path(
                        "video_dir/video0/Export_py/Video_pitchshift.mp4"
                    ),
                    "dialogue": [
                        {"role": "system", "content": "system message", "eval": False},
                        {"role": "assistant", "content": "a0", "eval": False},
                        {"role": "assistant", "content": "a1", "eval": False},
                        {"role": "user", "content": "u0", "eval": False},
                        {"role": "assistant", "content": "a2", "eval": True},
                    ],
                    "encoded_frames_path": Path("video_frame_dir_path/video0.pt"),
                }
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
            None,
            [
                {
                    "index": 0,
                    "video_id": "video0",
                    "video_path": Path(
                        "video_dir/video0/Export_py/Video_pitchshift.mp4"
                    ),
                    "dialogue": [
                        {"role": "system", "content": "system message", "eval": False},
                        {"role": "assistant", "content": "a0", "eval": False},
                        {"role": "assistant", "content": "a1", "eval": False},
                        {"role": "user", "content": "u0", "eval": False},
                        {"role": "assistant", "content": "a2", "eval": True},
                        {"role": "assistant", "content": "a3", "eval": True},
                        {"role": "assistant", "content": "a4", "eval": True},
                    ],
                },
                {
                    "index": 1,
                    "video_id": "video0",
                    "video_path": Path(
                        "video_dir/video0/Export_py/Video_pitchshift.mp4"
                    ),
                    "dialogue": [
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
                },
                {
                    "index": 2,
                    "video_id": "video1",
                    "video_path": Path(
                        "video_dir/video1/Export_py/Video_pitchshift.mp4"
                    ),
                    "dialogue": [
                        {"role": "system", "content": "system message", "eval": False},
                        {"role": "assistant", "content": "a0", "eval": False},
                        {"role": "assistant", "content": "a1", "eval": False},
                        {"role": "user", "content": "u0", "eval": False},
                        {"role": "assistant", "content": "a2", "eval": True},
                        {"role": "assistant", "content": "a3", "eval": True},
                    ],
                },
                {
                    "index": 3,
                    "video_id": "video1",
                    "video_path": Path(
                        "video_dir/video1/Export_py/Video_pitchshift.mp4"
                    ),
                    "dialogue": [
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
                },
            ],
        ),
    ],
)
def test_holo_assist_dataset_init(
    tmp_path: Path,
    original_anns: dict,
    video_frame_dir_path: Path | None,
    expected_items: list[dict],
) -> None:
    anns_file = tmp_path / "ann.json"
    anns_file.write_text(json.dumps(original_anns))

    dataset = HoloAssistDataset(
        anns_file,
        video_dir_path=Path("video_dir"),
        video_frame_dir_path=video_frame_dir_path,
    )
    assert list(iter(dataset)) == expected_items


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
    anns: dict[str, list[dict]], expected: list[tuple[str, list[dict]]]
) -> None:
    assert _convert_real_time_anns_to_datapoint(anns) == expected
