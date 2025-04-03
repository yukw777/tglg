from functools import partial

from real_time_vlm_benchmark.baseline_models.videollm_online_models.videollm_online import (
    RealTimeModel,
    VideoLLMOnlineModel,
)


def holo_assist_system_message(dialogue: list[dict], use_narration: bool) -> str:
    if use_narration:
        assert dialogue[0]["role"] == "system"
    return (
        (
            "A multimodal AI assistant is helping users with some activities. "
            "Below is their conversation, interleaved with the list of video frames received by the assistant. "
            "The assistant should give the user instructions and correct their mistakes. "
            f"Here's the summary of the activity: {dialogue[0]['content']}"
        )
        if use_narration
        else "A multimodal AI assistant is helping users with some activities. Below is their conversation, interleaved with the list of video frames received by the assistant."
    )


class VideoLLMOnlineHoloAssistModel(VideoLLMOnlineModel):
    def __init__(self, use_narration: bool = False, **kwargs) -> None:
        super().__init__(
            **kwargs,
            sys_msg_fn=partial(holo_assist_system_message, use_narration=use_narration),
        )


class RealTimeHoloAssistModel(RealTimeModel):
    def __init__(self, use_narration: bool = False, **kwargs) -> None:
        super().__init__(
            **kwargs,
            sys_msg_fn=partial(holo_assist_system_message, use_narration=use_narration),
        )
