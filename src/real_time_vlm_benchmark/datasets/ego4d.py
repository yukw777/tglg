import json
from pathlib import Path
from typing import Any, Callable

from real_time_vlm_benchmark.datasets.real_time import RealTimeDataset


def _convert_real_time_anns_to_datapoint(
    anns: list[dict],
) -> list[tuple[str, list[dict]]]:
    def _calc_end_time(content: str, start: float) -> float:
        num_words = len(content.split())
        # assume 150 wpm
        end = start + num_words / 150 * 60
        return end

    data: list[tuple[str, list[dict]]] = []
    for ann in anns:
        conversation = ann["conversation"]
        curr_user_utter: dict | None = None
        for i, utter in enumerate(conversation):
            if utter["role"] == "user":
                if utter["content"].startswith("(") and utter["content"].endswith(")"):
                    # some user utterances are user actions, e.g., (starts slicing cake), so we should skip this
                    continue
                # NOTE: curr_user_utter may already be set, but we overwrite since it's not followed by an assistant utterance.
                curr_user_utter = utter
                continue
            if curr_user_utter is None:
                # a standalone assistant utterance
                start = utter["time"]
                end = _calc_end_time(utter["content"], start)
                data.append(
                    (
                        ann["video_uid"],
                        [
                            {
                                "role": utter["role"],
                                "content": utter["content"],
                                "start": start,
                                "end": end,
                            }
                        ],
                    )
                )
            else:
                # an assistant answer to a user query
                # first user utterance
                user_start = curr_user_utter["time"]
                user_end = _calc_end_time(curr_user_utter["content"], user_start)

                # now the assistant response
                # typical latency in everyday conversation is under 0.3 seconds
                assistant_start = user_end + 0.3
                assistant_end = _calc_end_time(utter["content"], assistant_start)

                data.append(
                    (
                        ann["video_uid"],
                        [
                            {
                                "role": curr_user_utter["role"],
                                "content": curr_user_utter["content"],
                                "start": user_start,
                                "end": user_end,
                            },
                            {
                                "role": utter["role"],
                                "content": utter["content"],
                                "start": assistant_start,
                                "end": assistant_end,
                            },
                        ],
                    )
                )
                curr_user_utter = None
    return data


class Ego4dGoalStepDataset(RealTimeDataset):
    def __init__(
        self,
        video_dir_path: Path,
        ann_file_path: Path,
        video_frame_dir_path: Path | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        super().__init__()
        self.video_dir_path = video_dir_path
        self.video_frame_dir_path = video_frame_dir_path
        self._preprocessor = preprocessor
        with open(ann_file_path) as f:
            anns = json.load(f)
        self.data = _convert_real_time_anns_to_datapoint(anns)

    def __getitem__(self, index: int) -> dict:
        video_id, dialogue = self.data[index]
        datapoint = {
            "index": index,
            "video_id": video_id,
            "video_path": self.video_dir_path / f"{video_id}.mp4",
            "dialogue": dialogue,
        }
        if self.video_frame_dir_path is not None:
            datapoint["encoded_frames_path"] = (
                self.video_frame_dir_path / f"{video_id}.pt"
            )
        if self.preprocessor is not None:
            return self.preprocessor(datapoint)
        return datapoint

    def __len__(self) -> int:
        return len(self.data)
