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
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, sys_msg_fn=soccernet_system_message)


class RealTimeSoccerNetModel(RealTimeModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, sys_msg_fn=soccernet_system_message)
