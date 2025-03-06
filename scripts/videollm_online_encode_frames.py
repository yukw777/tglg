import sys
from pathlib import Path

import torch

# decord must be imported after torch
# https://github.com/dmlc/decord/issues/293
import decord  # isort: skip
from dataclasses import asdict

from accelerate import Accelerator
from accelerate.utils import set_seed, tqdm
from decord import VideoReader
from einops import rearrange
from jsonargparse import auto_cli
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.v2.functional import resize
from transformers import AutoModel
from videollm_online.models.arguments_live import get_args_class
from videollm_online.models.configuration_live import LiveConfigMixin
from videollm_online.models.vision_live import build_live_vision

from real_time_vlm_benchmark.baseline_models.videollm_online_models.holo_assist import (
    sample_frames_for_dialogue,
)
from real_time_vlm_benchmark.datasets.holo_assist import HoloAssistDataset


def run(
    dataset: HoloAssistDataset,
    results_dir: Path,
    version: str = "live1+",
    per_device_batch_size: int = 256,
    num_dataloader_workers: int = 4,
    start_idx: int | None = None,
    end_idx: int | None = None,
    random_seed: int = 42,
    torch_dtype: str = "bfloat16",
) -> int:
    set_seed(random_seed)

    # set up model
    args = get_args_class(version)()
    vision_config = LiveConfigMixin(**asdict(args))
    _, vision_encode = build_live_vision(vision_config)
    # Initialize the model manually in order to set torch_dtype
    vision_model = AutoModel.from_pretrained(
        vision_config.vision_pretrained, torch_dtype=getattr(torch, torch_dtype)
    ).vision_model

    # initialize accelerator
    # NOTE: accelerator has to be initialized after model initialization
    accelerator = Accelerator()
    vision_model.eval()
    vision_model.to(accelerator.device)

    # set up the preprocessor
    decord.bridge.set_bridge("torch")

    frame_resolution = args.frame_resolution
    max_num_frames = args.max_num_frames
    frame_fps = args.frame_fps

    def preprocess(datapoint: dict) -> dict[str, torch.Tensor]:
        vr = VideoReader(str(datapoint["video"]))
        dialogue = datapoint["dialogue"]

        # sample frames from max(0, end time - max_num_frames/frame_fps) to the end time of the last utterance at self.frame_fps
        end_time = dialogue[-1]["end"]
        start_time = max(0, end_time - max_num_frames / frame_fps)
        frame_idx = sample_frames_for_dialogue(
            start_time, end_time, vr.get_avg_fps(), frame_fps
        )
        frames = vr.get_batch(frame_idx)
        frames = rearrange(frames, "t h w c -> t c h w")
        frames = resize(frames, [frame_resolution] * 2)
        return {"index": torch.tensor(datapoint["index"]), "frames": frames}

    dataset.preprocessor = preprocess
    # set up data by figuring out indices to run inference on
    # first, get all the indices
    inference_indices = set(range(len(dataset)))

    # second, check results_dir for finished indices
    finished_idx: set[int] = set()
    if results_dir.exists():
        # get all the finished indices
        finished_idx = set(int(f.stem) for f in results_dir.iterdir())
    else:
        results_dir.mkdir(parents=True)

    # third, remove finished indices from inference indices
    inference_indices -= finished_idx

    # finally, take care of start_idx and end_idx
    # [start_idx, end_idx)
    if start_idx is not None:
        inference_indices = set(idx for idx in inference_indices if idx >= start_idx)
    if end_idx is not None:
        inference_indices = set(idx for idx in inference_indices if idx < end_idx)

    def collate(datapoints: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "index": torch.stack([dp["index"] for dp in datapoints]),
            "frames": torch.cat([dp["frames"] for dp in datapoints]),
            "num_frames": torch.tensor([dp["frames"].size(0) for dp in datapoints]),
        }

    with accelerator.split_between_processes(
        sorted(inference_indices)
    ) as per_process_indices:
        dataloader = DataLoader(
            Subset(dataset, per_process_indices),
            batch_size=per_device_batch_size,
            num_workers=num_dataloader_workers,
            pin_memory=True,
            collate_fn=collate,
        )
        failure = torch.tensor(False, device=accelerator.device)
        for batch in tqdm(dataloader, desc="Encode"):
            try:
                with torch.inference_mode():
                    batch_encoded_frames = vision_encode(
                        vision_model, batch["frames"].to(accelerator.device)
                    )
            except Exception as e:
                accelerator.print(
                    f"[rank {accelerator.process_index}] Exception raised, skipping batch: {e}"
                )
                failure = torch.tensor(True, device=accelerator.device)
                continue
            for index, encoded_frames in zip(
                batch["index"],
                batch_encoded_frames.split(batch["num_frames"].tolist()),
                strict=True,
            ):
                torch.save(
                    encoded_frames.to(torch.device("cpu"), getattr(torch, torch_dtype)),
                    results_dir / f"{index.item()}.pt",
                )
    success = (~torch.any(accelerator.gather(failure))).item()
    accelerator.end_training()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(auto_cli(run, as_positional=False))
