import json
from pathlib import Path
from typing import Any, Callable

from real_time_vlm_benchmark.datasets.real_time import RealTimeDataset


def _convert_real_time_anns_to_datapoint(
    anns: dict[str, list[dict]], tolerance: float = 5
) -> list[tuple[str, list[dict]]]:
    data: list[tuple[str, list[dict]]] = []
    for video_id, dialogue in anns.items():
        curr_segment: list[dict] = [dialogue[0]]
        i = 1
        while i < len(dialogue):
            if curr_segment[-1]["end"] + tolerance > dialogue[i]["start"]:
                # if the current utterance's start time is within tolerance seconds of
                # the end time of the last utterance of the current segment, add.
                curr_segment.append(dialogue[i])
            else:
                # otherwise, start a new segment
                data.append((video_id, curr_segment))
                curr_segment = [dialogue[i]]
            i += 1
        # take care of stragglers
        if len(curr_segment) > 0:
            data.append((video_id, curr_segment))
    return data


class SoccerNetDataset(RealTimeDataset):
    def __init__(
        self,
        video_dir_path: str,
        ann_file_path: str,
        video_frame_dir_path: str | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.video_frame_dir_path = (
            Path(video_frame_dir_path) if video_frame_dir_path is not None else None
        )
        self._preprocessor = preprocessor
        with open(ann_file_path) as f:
            anns = json.load(f)
        self.data = _convert_real_time_anns_to_datapoint(anns)
        self.video_paths: dict[str, Path] = {}
        for video_path in Path(video_dir_path).glob("**/*.mkv"):
            self.video_paths[f"{video_path.parts[-2]}/{video_path.stem}"] = video_path

    def __getitem__(self, index: int) -> dict:
        video_id, dialogue = self.data[index]
        datapoint = {
            "index": index,
            "video_id": video_id,
            "video_path": self.video_paths[video_id],
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
