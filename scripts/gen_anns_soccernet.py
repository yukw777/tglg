import json
from pathlib import Path

from jsonargparse import auto_cli

from real_time_vlm_benchmark.datasets.soccernet.utils import convert_pbp_annotated


def main(pbp_annotated_dir: str, output_file: str) -> None:
    anns = {}
    for pbp_annotated_path in Path(pbp_annotated_dir).glob("**/*.json"):
        with open(pbp_annotated_path) as f:
            pbp_annotated = json.load(f)
        utters = convert_pbp_annotated(pbp_annotated)
        if len(utters) == 0:
            continue
        anns[
            "/".join(
                [
                    pbp_annotated_path.parts[-2],
                    pbp_annotated_path.stem.replace("_annotated", ""),
                ]
            )
        ] = utters
    with open(output_file, "w") as f:
        json.dump(anns, f, indent=4)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
