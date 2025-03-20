from functools import partial

from real_time_vlm_benchmark.baseline_models.videollm_online_models import (
    VideoLLMOnlineModel,
)


def _construct_system_message(dialogue: list[dict], use_narration: bool) -> str:
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
    def __init__(
        self,
        version: str = "live1+",
        checkpoint: str = "chenjoya/videollm-online-8b-v1plus",
        frame_token_interval_threshold: float = 0.725,
        use_narration: bool = False,
        show_progress: bool = False,
        set_vision_inside: bool = False,
    ) -> None:
        super().__init__(
            version=version,
            checkpoint=checkpoint,
            frame_token_interval_threshold=frame_token_interval_threshold,
            show_progress=show_progress,
            set_vision_inside=set_vision_inside,
            sys_msg_fn=partial(_construct_system_message, use_narration=use_narration),
        )
