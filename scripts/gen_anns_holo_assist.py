import json
from pathlib import Path

from real_time_vlm_benchmark.datasets.holo_assist import convert_holo_assist


def main(
    holo_assist_video_dir: str, holo_assist_anns_file: str, output_file: str
) -> None:
    with open(holo_assist_anns_file) as f:
        holo_assist_anns = json.load(f)

    anns = convert_holo_assist(holo_assist_anns)

    # Filter out annotations for non-existent videos
    videos = list(anns.keys())
    video_dir = Path(holo_assist_video_dir)
    for video in videos:
        if not (video_dir / video).exists():
            print(f"Warning: Video {video} doesn't exist")
            del anns[video]

    with open(output_file, "w") as f:
        json.dump(anns, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--holo_assist_video_dir", required=True)
    parser.add_argument("--holo_assist_anns_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    main(args.holo_assist_video_dir, args.holo_assist_anns_file, args.output_file)
