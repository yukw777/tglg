from typing import Any

import pytest

from real_time_vlm_benchmark.datasets.holo_assist import convert_holo_assist


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
    ],
)
def test_convert_holo_assist(
    holo_assist_anns: list[dict], expected: dict[str, list[dict[str, Any]]]
) -> None:
    assert convert_holo_assist(holo_assist_anns) == expected
