import json
from pathlib import Path

import torch
import yaml
from jsonargparse import auto_cli
from tqdm import tqdm

from real_time_vlm_benchmark.datasets.soccernet.pbp_model import (
    PBPCommentaryBiLSTMClassifier,
    PBPCommentaryDataModule,
)


def main(
    pbp_classifier_config_path: str,
    pbp_classifier_ckpt_path: str,
    transcription_dir: str,
    output_dir: str,
    batch_size: int = 128,
    device: str = "cuda",
) -> None:
    # load config
    with open(pbp_classifier_config_path) as f:
        dm_config = yaml.safe_load(f)["data"]
    dm_config["predict_batch_size"] = batch_size

    # load model
    model = PBPCommentaryBiLSTMClassifier.load_from_checkpoint(pbp_classifier_ckpt_path)
    model.eval()
    model.to(device)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Figure out the progress
    completed = set(
        tuple(classified_file.parts[-2:])
        for classified_file in output_dir_path.glob("**/*.json")
    )
    transcription_files = []
    for transcription_file in Path(transcription_dir).glob("**/*.json"):
        if tuple(transcription_file.parts[-2:]) not in completed:
            transcription_files.append(transcription_file)

    for transcription_file in tqdm(transcription_files, desc="Transcriptions"):
        dm_config["predict_transcript_file"] = str(transcription_file)
        dm = PBPCommentaryDataModule(**dm_config)
        try:
            dm.setup("predict")
        except ValueError:
            # no segments so skip
            continue
        segments = []
        for i, batch in tqdm(
            enumerate(dm.predict_dataloader()), desc="Classifying", leave=False
        ):
            batch["sent_seq"] = batch["sent_seq"].to(device)
            with torch.inference_mode():
                results = model.predict_step(batch, i)
            segments.extend(results)

        output_file_dir_path = output_dir_path / transcription_file.parts[-2]
        output_file_dir_path.mkdir(parents=True, exist_ok=True)
        with open(output_file_dir_path / transcription_file.name, "w") as f:
            json.dump({"segments": segments}, f, indent=4)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
