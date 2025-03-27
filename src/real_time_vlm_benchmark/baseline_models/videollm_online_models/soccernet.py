from real_time_vlm_benchmark.baseline_models.videollm_online_models.videollm_online import (
    RealTimeModel,
    VideoLLMOnlineModel,
)


def soccernet_system_message(dialogue: list[dict]) -> str:
    return (
        "You're a soccer play-by-play commentator. "
        "Blow are your commentaries, interleaved with the list of video frames from the soccer match. "
        "You should give play-by-play commentaries based on the video frames."
    )


class VideoLLMOnlineSoccerNetModel(VideoLLMOnlineModel):
    def __init__(
        self,
        version: str = "live1+",
        checkpoint: str = "chenjoya/videollm-online-8b-v1plus",
        frame_token_interval_threshold: float = 0.725,
        show_progress: bool = False,
        set_vision_inside: bool = False,
    ) -> None:
        super().__init__(
            version=version,
            checkpoint=checkpoint,
            frame_token_interval_threshold=frame_token_interval_threshold,
            show_progress=show_progress,
            set_vision_inside=set_vision_inside,
            sys_msg_fn=soccernet_system_message,
        )


class RealTimeSoccerNetModel(RealTimeModel):
    def __init__(
        self,
        version: str = "live1+",
        checkpoint: str = "chenjoya/videollm-online-8b-v1plus",
        frame_token_interval_threshold: float = 0.725,
        show_progress: bool = False,
        set_vision_inside: bool = False,
    ) -> None:
        super().__init__(
            version=version,
            checkpoint=checkpoint,
            frame_token_interval_threshold=frame_token_interval_threshold,
            show_progress=show_progress,
            set_vision_inside=set_vision_inside,
            sys_msg_fn=soccernet_system_message,
        )
