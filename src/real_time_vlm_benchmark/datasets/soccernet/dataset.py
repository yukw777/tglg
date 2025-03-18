import json
from pathlib import Path
from typing import Any, Callable

from torch.utils.data import Dataset

from real_time_vlm_benchmark.datasets.utils import convert_real_time_anns_to_datapoint


class SoccerNetDataset(Dataset):
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
        self.preprocessor = preprocessor
        with open(ann_file_path) as f:
            anns = json.load(f)
        self.data = convert_real_time_anns_to_datapoint(anns)
        self.video_paths: dict[str, Path] = {}
        for video_path in Path(video_dir_path).glob("**/*.mkv"):
            self.video_paths[f"{video_path.parts[-2]}/{video_path.stem}"] = video_path

    def __getitem__(self, index: int) -> dict:
        video, dialogue = self.data[index]
        datapoint = {
            "index": index,
            "video": self.video_paths[video],
            "dialogue": dialogue,
        }
        if self.video_frame_dir_path is not None:
            datapoint["video_frame"] = self.video_frame_dir_path / f"{index}.pt"
        if self.preprocessor is not None:
            return self.preprocessor(datapoint)
        return datapoint

    def __len__(self) -> int:
        return len(self.data)
