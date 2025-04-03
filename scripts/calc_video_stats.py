import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from decord import VideoReader
from jsonargparse import auto_cli
from torch.utils.data import Dataset
from tqdm import tqdm


def get_stats(video_path: Path) -> dict:
    vr = VideoReader(str(video_path))
    return {"fps": vr.get_avg_fps(), "num_frames": len(vr)}


def main(real_time_dataset: Dataset, out_file: Path) -> None:
    videos = set(
        (datapoint["video_id"], datapoint["video_path"])
        for datapoint in iter(real_time_dataset)
    )
    video_stats = {}
    with ThreadPoolExecutor() as executor:
        future_to_video_id = {
            executor.submit(get_stats, video_path): video_id
            for video_id, video_path in videos
        }
        for future in tqdm(
            as_completed(future_to_video_id), total=len(future_to_video_id)
        ):
            video_id = future_to_video_id[future]
            video_stats[video_id] = future.result()

    with open(out_file, "w") as f:
        json.dump(video_stats, f, indent=4)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
