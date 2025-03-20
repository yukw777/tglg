import multiprocessing as mp
import sys
from multiprocessing.managers import BaseManager
from pathlib import Path
from queue import Queue

import torch

# decord must be imported after torch
# https://github.com/dmlc/decord/issues/293
import decord  # isort: skip
from dataclasses import asdict

from accelerate import Accelerator
from accelerate.utils import set_seed
from decord import VideoReader
from einops import rearrange
from jsonargparse import auto_cli
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)
from torchvision.transforms.v2.functional import resize
from tqdm import tqdm
from transformers import AutoModel
from videollm_online.models.arguments_live import get_args_class
from videollm_online.models.configuration_live import LiveConfigMixin
from videollm_online.models.vision_live import build_live_vision

from real_time_vlm_benchmark.baseline_models.utils.sample import QueueSampler
from real_time_vlm_benchmark.datasets.utils import convert_to_frame_dataset


class FrameDataset(Dataset):
    def __init__(self, data: list[dict], frame_resolution: int) -> None:
        super().__init__()
        self.data = data
        self.frame_resolution = frame_resolution

    def __getitem__(self, index: int) -> dict:
        datapoint = self.data[index]
        vr = VideoReader(str(datapoint["video_path"]))
        frames = vr.get_batch(datapoint["frame_idx"])
        frames = rearrange(frames, "t h w c -> t c h w")
        frames = resize(frames, [self.frame_resolution] * 2)
        return {
            "video_id": datapoint["video_id"],
            "frame_idx": datapoint["frame_idx"],
            "frames": frames,
        }

    def __len__(self) -> int:
        return len(self.data)


def run(
    dataset: Dataset,
    results_dir: Path,
    version: str = "live1+",
    per_device_num_frame: int = 512,
    per_device_num_video: int = 2,
    num_dataloader_workers: int = 4,
    start_idx: int | None = None,
    end_idx: int | None = None,
    random_seed: int = 42,
    torch_dtype: str = "bfloat16",
    mp_manager_ip_addr: str = "",
    mp_manager_port: int = 12345,
    mp_manager_auth_key: bytes = b"password",
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

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(dataset)  # type: ignore

    frame_data = convert_to_frame_dataset(
        Subset(dataset, list(range(start_idx, end_idx))),
        args.frame_fps,
        max_num_frames=args.max_num_frames,
    )

    # filter out finished frame indices
    filtered_frame_data = []
    for i in range(len(frame_data)):
        datapoint = frame_data[i]
        frame_id_set = set(datapoint["frame_idx"].tolist())
        finished_id_set: set[int] = set()
        frames_dir = results_dir / datapoint["video_id"]
        frames_dir.mkdir(parents=True, exist_ok=True)
        for f in (frames_dir).iterdir():
            finished_id_set.add(int(f.stem))
        frame_idx = torch.tensor(sorted(frame_id_set - finished_id_set))
        if frame_idx.size(0) == 0:
            continue
        filtered_frame_data.append(
            {
                "video_id": datapoint["video_id"],
                "video_path": datapoint["video_path"],
                "frame_idx": frame_idx,
            }
        )

    # initialize accelerator
    # NOTE: accelerator has to be initialized after model initialization
    accelerator = Accelerator()

    # set up the queues
    class QueueManager(BaseManager):
        pass

    class ProgressBarProcess(mp.Process):
        def __init__(self, progress_queue: mp.Queue, total: int) -> None:
            self.progress_queue = progress_queue
            self.progress_bar = tqdm(total=total, desc="Encode")
            super().__init__()

        def run(self) -> None:
            while True:
                progress = self.progress_queue.get()
                if progress is None:
                    self.progress_bar.close()
                    return
                self.progress_bar.update(progress)

    if accelerator.is_main_process:
        queue: Queue[int] = Queue()
        for i in range(len(filtered_frame_data)):
            queue.put(i)
        QueueManager.register("get_queue", lambda: queue)
        # NOTE: we have to use mp.Queue(), not the regular Queue b/c
        # the progress bar process is a local process.
        progress_queue: mp.Queue[int | None] = mp.Queue()
        QueueManager.register("get_progress_queue", lambda: progress_queue)
        manager = QueueManager(
            address=(mp_manager_ip_addr, mp_manager_port),
            authkey=mp_manager_auth_key,
        )
        manager.start()
        progress_bar_proc = ProgressBarProcess(progress_queue, len(filtered_frame_data))
        progress_bar_proc.start()
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        QueueManager.register("get_queue")
        QueueManager.register("get_progress_queue")
        manager = QueueManager(
            address=(mp_manager_ip_addr, mp_manager_port),
            authkey=mp_manager_auth_key,
        )
        manager.connect()
        queue = manager.get_queue()  # type: ignore
        progress_queue = manager.get_progress_queue()  # type: ignore
    accelerator.wait_for_everyone()

    frame_dataset = FrameDataset(filtered_frame_data, args.frame_resolution)

    # set up the model
    vision_model.eval()
    vision_model.to(accelerator.device)

    # set up the preprocessor
    decord.bridge.set_bridge("torch")

    # set up the dataloader
    def collate(datapoints: list[dict]) -> dict:
        return {
            "video_id": [dp["video_id"] for dp in datapoints],
            "frame_idx": [dp["frame_idx"].tolist() for dp in datapoints],
            "frames": torch.cat([dp["frames"] for dp in datapoints]),
        }

    dataloader = DataLoader(
        frame_dataset,
        batch_size=per_device_num_video,
        num_workers=num_dataloader_workers,
        pin_memory=True,
        collate_fn=collate,
        sampler=QueueSampler(queue),
    )
    failure = torch.tensor(False, device=accelerator.device)
    for batch in dataloader:
        try:
            batch_frames = batch["frames"].split(per_device_num_frame)
            batch_encoded_frames_list = []
            with torch.inference_mode():
                for frames in batch_frames:
                    encoded_frames = vision_encode(
                        vision_model, frames.to(accelerator.device)
                    )
                    batch_encoded_frames_list.append(encoded_frames)
        except Exception as e:
            print(
                f"[rank {accelerator.process_index}] Exception raised for batch {batch['video_id'].tolist()}. Skipping: {e}"
            )
            failure = torch.tensor(True, device=accelerator.device)
            continue
        batch_encoded_frames = torch.cat(batch_encoded_frames_list)
        for video_id, frame_idx, encoded_frames in zip(
            batch["video_id"],
            batch["frame_idx"],
            batch_encoded_frames.split(
                [len(frame_idx) for frame_idx in batch["frame_idx"]]
            ),
            strict=True,
        ):
            for frame_id, encoded_frame in zip(frame_idx, encoded_frames, strict=True):
                torch.save(
                    encoded_frame.to(torch.device("cpu"), getattr(torch, torch_dtype)),
                    results_dir / video_id / f"{frame_id}.pt",
                )
        progress_queue.put(len(batch["video_id"]))
    success = (~torch.any(accelerator.gather(failure))).item()
    # signal the progress bar process to exit
    if accelerator.is_main_process:
        progress_queue.put(None)
    accelerator.end_training()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(auto_cli(run, as_positional=False))
