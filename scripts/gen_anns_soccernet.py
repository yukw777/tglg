import json
from pathlib import Path

from jsonargparse import auto_cli
from torch.utils.data import random_split

from real_time_vlm_benchmark.datasets.soccernet.utils import convert_pbp_annotated


def main(
    pbp_annotated_dir: str,
    output_file_prefix: str,
    output_dir: str,
    val_test_ratio: tuple[float, float] = (0.2, 0.2),
) -> None:
    anns = {}
    for pbp_annotated_path in Path(pbp_annotated_dir).glob("**/*.json"):
        with open(pbp_annotated_path) as f:
            pbp_annotated = json.load(f)
        utters = convert_pbp_annotated(pbp_annotated)
        if len(utters) <= 20:
            # This most likely means the matches weren't actually commentated.
            continue
        anns[
            "/".join(
                [
                    pbp_annotated_path.parts[-2],
                    pbp_annotated_path.stem.replace("_annotated", ""),
                ]
            )
        ] = utters

    # split train, val and test
    videos = list(anns.keys())
    val_ratio, test_ratio = val_test_ratio
    train_videos, test_videos = random_split(videos, (1 - test_ratio, test_ratio))  # type: ignore
    train_videos, val_videos = random_split(train_videos, (1 - val_ratio, val_ratio))

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    all_file = output_dir_path / f"{output_file_prefix}.json"
    with open(all_file, "w") as f:
        json.dump({video: anns[video] for video in iter(videos)}, f, indent=4)

    train_file = output_dir_path / f"{output_file_prefix}_train.json"
    with open(train_file, "w") as f:
        json.dump({video: anns[video] for video in iter(train_videos)}, f, indent=4)

    val_file = output_dir_path / f"{output_file_prefix}_val.json"
    with open(val_file, "w") as f:
        json.dump({video: anns[video] for video in iter(val_videos)}, f, indent=4)

    test_file = output_dir_path / f"{output_file_prefix}_test.json"
    with open(test_file, "w") as f:
        json.dump({video: anns[video] for video in iter(test_videos)}, f, indent=4)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
