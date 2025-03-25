from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import torch
from jsonargparse import auto_cli
from tqdm import tqdm


def combine(
    encoded_frames_dir_path: Path, output_dir_path: Path, video_dir: Path
) -> None:
    frame_idx = []
    frame_embeds_list = []
    video_id = str(video_dir.relative_to(encoded_frames_dir_path))
    for frame_file in video_dir.iterdir():
        frame_idx.append(int(frame_file.stem))
        frame_embeds_list.append(torch.load(frame_file))
    frame_embeds = torch.stack(frame_embeds_list)
    frames_dict = dict(zip(frame_idx, frame_embeds))
    output_path = output_dir_path / f"{video_id}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(frames_dict, output_path)


def main(encoded_frames_dir: str, output_dir: str) -> None:
    encoded_frames_dir_path = Path(encoded_frames_dir)
    video_dirs = [
        p
        for p in tqdm(encoded_frames_dir_path.glob("**/*"), desc="Finding video dirs")
        if not p.name.endswith(".pt")
    ]
    output_dir_path = Path(output_dir)
    combine_fn = partial(combine, encoded_frames_dir_path, output_dir_path)
    with ThreadPoolExecutor() as executor:
        future_to_video_dir = {
            executor.submit(combine_fn, video_dir): video_dir
            for video_dir in video_dirs
        }
        for future in tqdm(
            as_completed(future_to_video_dir),
            total=len(future_to_video_dir),
            desc="Combining",
        ):
            video_dir = future_to_video_dir[future]
            try:
                future.result()
            except Exception as e:
                print(f"Exception for {video_dir}: {e}")


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
