from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import torch
from jsonargparse import auto_cli
from tqdm import tqdm

from real_time_vlm_benchmark.datasets import RealTimeDataset


def combine(
    encoded_frames_dir_path: Path, output_dir_path: Path, video_id: str
) -> None:
    frame_idx = []
    frame_embeds_list = []
    video_dir = encoded_frames_dir_path / video_id
    for frame_file in video_dir.iterdir():
        frame_idx.append(int(frame_file.stem))
        frame_embeds_list.append(torch.load(frame_file))
    frame_embeds = torch.stack(frame_embeds_list)
    frames_dict = dict(zip(frame_idx, frame_embeds))
    output_path = output_dir_path / f"{video_id}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(frames_dict, output_path)


def main(
    real_time_dataset: RealTimeDataset, encoded_frames_dir: str, output_dir: str
) -> None:
    encoded_frames_dir_path = Path(encoded_frames_dir)
    video_ids = set(d["video_id"] for d in iter(real_time_dataset))
    output_dir_path = Path(output_dir)
    combine_fn = partial(combine, encoded_frames_dir_path, output_dir_path)
    with ThreadPoolExecutor() as executor:
        future_to_video_id = {
            executor.submit(combine_fn, video_id): video_id for video_id in video_ids
        }
        for future in tqdm(
            as_completed(future_to_video_id),
            total=len(future_to_video_id),
            desc="Combining",
        ):
            video_id = future_to_video_id[future]
            try:
                future.result()
            except Exception as e:
                print(f"Exception for {video_id}: {e}")


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
