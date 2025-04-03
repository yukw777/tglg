import math

import pytest

from real_time_vlm_benchmark.datasets.ego4d import _convert_real_time_anns_to_datapoint


@pytest.mark.parametrize(
    "anns,expected",
    [
        (
            [
                {
                    "video_uid": "f7a0beb6-b220-40c0-a72d-ec4b79134a73",
                    "conversation": [
                        {
                            "role": "user",
                            "content": "What are the prerequisites for the next task?",
                            "time": 138.6,
                        },
                        {
                            "role": "assistant",
                            "content": "Before proceeding, please make sure you have sliced the tomatoes. We've already added pepper, salt, and sauce to them.",
                            "time": 138.6,
                        },
                    ],
                }
            ],
            [
                (
                    "f7a0beb6-b220-40c0-a72d-ec4b79134a73",
                    [
                        {
                            "role": "user",
                            "content": "What are the prerequisites for the next task?",
                            "start": 138.6,
                            "end": 141.8,
                        },
                        {
                            "role": "assistant",
                            "content": "Before proceeding, please make sure you have sliced the tomatoes. We've already added pepper, salt, and sauce to them.",
                            "start": 142.1,
                            "end": 149.7,
                        },
                    ],
                )
            ],
        ),
        (
            [
                {
                    "video_uid": "f7a0beb6-b220-40c0-a72d-ec4b79134a73",
                    "conversation": [
                        {
                            "role": "assistant",
                            "content": "You've just added salt to the sliced tomatoes. The next step is to add sauce.",
                            "time": 149.5,
                        }
                    ],
                }
            ],
            [
                (
                    "f7a0beb6-b220-40c0-a72d-ec4b79134a73",
                    [
                        {
                            "role": "assistant",
                            "content": "You've just added salt to the sliced tomatoes. The next step is to add sauce.",
                            "start": 149.5,
                            "end": 155.5,
                        },
                    ],
                )
            ],
        ),
        (
            [
                {
                    "video_uid": "f7a0beb6-b220-40c0-a72d-ec4b79134a73",
                    "conversation": [
                        {
                            "role": "user",
                            "content": "What are the prerequisites for the next task?",
                            "time": 138.6,
                        },
                        {
                            "role": "assistant",
                            "content": "Before proceeding, please make sure you have sliced the tomatoes. We've already added pepper, salt, and sauce to them.",
                            "time": 138.6,
                        },
                        {
                            "role": "assistant",
                            "content": "You've just added salt to the sliced tomatoes. The next step is to add sauce.",
                            "time": 149.5,
                        },
                    ],
                },
                {
                    "video_uid": "5cdf77b8-7bf8-421b-99b6-19fa6429aeb4",
                    "conversation": [
                        {
                            "role": "user",
                            "content": "What are the resources I'm using right now?",
                            "time": 557.0,
                        },
                        {
                            "role": "assistant",
                            "content": "You're currently in the kitchen, and I see that you're using various kitchen items, such as utensils and appliances. You're also using your phone.",
                            "time": 557.0,
                        },
                        {
                            "role": "user",
                            "content": "How does what I should do next impact the bigger picture?",
                            "time": 712.3,
                        },
                        {
                            "role": "assistant",
                            "content": "Based on our previous conversations, it seems that you're preparing a meal that involves rice and coconut. To complete the dish, you'll need to grate the coconut. I recommend focusing on that task next.",
                            "time": 712.3,
                        },
                        {
                            "role": "assistant",
                            "content": "You're now preparing coconut. Would you like some tips on how to do it efficiently?",
                            "time": 741.0,
                        },
                    ],
                },
            ],
            [
                (
                    "f7a0beb6-b220-40c0-a72d-ec4b79134a73",
                    [
                        {
                            "role": "user",
                            "content": "What are the prerequisites for the next task?",
                            "start": 138.6,
                            "end": 141.8,
                        },
                        {
                            "role": "assistant",
                            "content": "Before proceeding, please make sure you have sliced the tomatoes. We've already added pepper, salt, and sauce to them.",
                            "start": 142.1,
                            "end": 149.7,
                        },
                    ],
                ),
                (
                    "f7a0beb6-b220-40c0-a72d-ec4b79134a73",
                    [
                        {
                            "role": "assistant",
                            "content": "You've just added salt to the sliced tomatoes. The next step is to add sauce.",
                            "start": 149.5,
                            "end": 155.5,
                        },
                    ],
                ),
                (
                    "5cdf77b8-7bf8-421b-99b6-19fa6429aeb4",
                    [
                        {
                            "role": "user",
                            "content": "What are the resources I'm using right now?",
                            "start": 557.0,
                            "end": 560.2,
                        },
                        {
                            "role": "assistant",
                            "content": "You're currently in the kitchen, and I see that you're using various kitchen items, such as utensils and appliances. You're also using your phone.",
                            "start": 560.5,
                            "end": 570.1,
                        },
                    ],
                ),
                (
                    "5cdf77b8-7bf8-421b-99b6-19fa6429aeb4",
                    [
                        {
                            "role": "user",
                            "content": "How does what I should do next impact the bigger picture?",
                            "start": 712.3,
                            "end": 716.7,
                        },
                        {
                            "role": "assistant",
                            "content": "Based on our previous conversations, it seems that you're preparing a meal that involves rice and coconut. To complete the dish, you'll need to grate the coconut. I recommend focusing on that task next.",
                            "start": 717,
                            "end": 730.6,
                        },
                    ],
                ),
                (
                    "5cdf77b8-7bf8-421b-99b6-19fa6429aeb4",
                    [
                        {
                            "role": "assistant",
                            "content": "You're now preparing coconut. Would you like some tips on how to do it efficiently?",
                            "start": 741.0,
                            "end": 747,
                        },
                    ],
                ),
            ],
        ),
    ],
)
def test_convert_real_time_anns_to_datapoint(
    anns: list[dict], expected: list[tuple[str, list[dict]]]
) -> None:
    datapoints = _convert_real_time_anns_to_datapoint(anns)
    assert len(datapoints) == len(expected)
    for (vid, conv), (exp_vid, exp_conv) in zip(datapoints, expected):
        assert vid == exp_vid
        assert len(conv) == len(exp_conv)
        for utter, exp_utter in zip(conv, exp_conv):
            assert utter["role"] == exp_utter["role"]
            assert utter["content"] == exp_utter["content"]
            assert math.isclose(utter["start"], exp_utter["start"])
            assert math.isclose(utter["end"], exp_utter["end"])
