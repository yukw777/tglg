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


def main(real_time_dataset: Dataset, results_dir: Path, out_file: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    videos = set(
        (datapoint["video_id"], datapoint["video_path"])
        for datapoint in iter(real_time_dataset)
    )
    completed_videos = set(f.stem for f in results_dir.glob("**/*.json"))
    videos -= completed_videos
    with ThreadPoolExecutor() as executor:
        future_to_video_id = {
            executor.submit(get_stats, video_path): video_id
            for video_id, video_path in videos
        }
        for future in tqdm(
            as_completed(future_to_video_id), total=len(future_to_video_id)
        ):
            video_id = future_to_video_id[future]
            with open(results_dir / f"{video_id}.json", "w") as f:
                json.dump(future.result(), f)

    video_stats = {}
    for video_stat_file in results_dir.glob("**/*.json"):
        with open(video_stat_file) as f:
            stats = json.load(f)
            video_stats[video_stat_file.stem] = stats
    with open(out_file, "w") as f:
        json.dump(video_stats, f, indent=4)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
