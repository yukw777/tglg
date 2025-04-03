import json
import multiprocessing as mp
import sys
from multiprocessing.managers import BaseManager
from pathlib import Path

import torch

# decord must be imported after torch
# https://github.com/dmlc/decord/issues/293
import decord  # isort: skip
from dataclasses import asdict

from accelerate import Accelerator
from accelerate.utils import set_seed
from decord import VideoReader
from einops import rearrange
from filelock import FileLock
from jsonargparse import auto_cli
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)
from torchvision.transforms.v2.functional import resize
from tqdm import tqdm
from transformers import AutoModel
from video_reader import PyVideoReader
from videollm_online.models.arguments_live import get_args_class
from videollm_online.models.configuration_live import LiveConfigMixin
from videollm_online.models.vision_live import build_live_vision

from real_time_vlm_benchmark.datasets import Ego4dGoalStepDataset
from real_time_vlm_benchmark.datasets.utils import chunked, convert_to_frame_dataset


class FrameDataset(Dataset):
    def __init__(
        self, data: list[dict], frame_resolution: int, use_decord: bool
    ) -> None:
        super().__init__()
        self.data = data
        self.frame_resolution = frame_resolution
        self.use_decord = use_decord

    def __getitem__(self, index: int) -> dict:
        datapoint = self.data[index]
        if self.use_decord:
            decord.bridge.set_bridge("torch")
            vr = VideoReader(str(datapoint["video_path"]))
            frames = vr.get_batch(datapoint["frame_idx"])
        else:
            # NOTE: video_reader-rs supports resizing at decoding, which helps keeping the memory usage low.
            vr = PyVideoReader(
                str(datapoint["video_path"]), resize_shorter_side=self.frame_resolution
            )
            frames = torch.from_numpy(vr.get_batch(datapoint["frame_idx"]))
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
    video_stats_file: Path,
    results_dir: Path,
    version: str = "live1+",
    frame_chunk_size: int = 512,
    per_device_batch_size: int = 2,
    num_dataloader_workers: int = 4,
    start_idx: int | None = None,
    end_idx: int | None = None,
    random_seed: int = 42,
    torch_dtype: str = "bfloat16",
    mp_manager_ip_addr: str = "",
    mp_manager_port: int = 12345,
    mp_manager_auth_key: bytes = b"password",
    use_decord: bool = True,
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

    with open(video_stats_file) as f:
        video_stats = json.load(f)
    if isinstance(dataset, Ego4dGoalStepDataset):
        # NOTE: Special handling for Ego4dGoalStepDataset
        # Some dialogues in Ego4dGoalStepDataset go beyond the video duration,
        # so let's filter them out by passing video_stats.
        dataset = Ego4dGoalStepDataset(
            dataset.video_dir_path,
            dataset.ann_file_path,
            video_frame_dir_path=dataset.video_frame_dir_path,
            video_stats=video_stats,
        )

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(dataset)  # type: ignore

    frame_data = convert_to_frame_dataset(
        Subset(dataset, list(range(start_idx, end_idx))),
        video_stats,
        args.frame_fps,
        max_num_frames=args.max_num_frames,
    )

    # filter out finished frame indices
    unchunked_filtered_frame_data = []
    for i in tqdm(range(len(frame_data)), desc="Filter Finished Indices"):
        datapoint = frame_data[i]
        frame_id_set = set(datapoint["frame_idx"].tolist())
        finished_id_set: set[int] = set()
        frames_file = results_dir / f"{datapoint['video_id']}.pt"
        if frames_file.exists():
            finished_frames_dict = torch.load(frames_file)
            finished_id_set.update(finished_frames_dict.keys())
        frame_idx = sorted(frame_id_set - finished_id_set)
        if len(frame_idx) == 0:
            continue
        unchunked_filtered_frame_data.append(
            {
                "video_id": datapoint["video_id"],
                "video_path": datapoint["video_path"],
                "frame_idx": frame_idx,
            }
        )

    # chunk frames to avoid decoding a large number of frames at once
    filtered_frame_data = []
    for datapoint in tqdm(unchunked_filtered_frame_data, desc="Chunk frames"):
        for frame_idx in chunked(datapoint["frame_idx"], frame_chunk_size):
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
        QueueManager.register("get_progress_queue")
        manager = QueueManager(
            address=(mp_manager_ip_addr, mp_manager_port),
            authkey=mp_manager_auth_key,
        )
        manager.connect()
        progress_queue = manager.get_progress_queue()  # type: ignore

    accelerator.wait_for_everyone()

    frame_dataset = FrameDataset(filtered_frame_data, args.frame_resolution, use_decord)

    # set up the model
    vision_model.eval()
    vision_model.to(accelerator.device)

    def collate(datapoints: list[dict]) -> dict:
        return {
            "video_id": [dp["video_id"] for dp in datapoints],
            "frame_idx": [dp["frame_idx"] for dp in datapoints],
            "frames": torch.cat([dp["frames"] for dp in datapoints]),
        }

    with accelerator.split_between_processes(
        sorted(range(len(frame_dataset)))
    ) as per_process_idx:
        dataloader = DataLoader(
            Subset(frame_dataset, per_process_idx),
            batch_size=per_device_batch_size,
            num_workers=num_dataloader_workers,
            pin_memory=True,
            collate_fn=collate,
        )
        failure = torch.tensor(False, device=accelerator.device)
        for batch in dataloader:
            try:
                with torch.inference_mode():
                    batch_encoded_frames = vision_encode(
                        vision_model, batch["frames"].to(accelerator.device)
                    )
            except Exception as e:
                print(
                    f"[rank {accelerator.process_index}] Exception raised for batch {batch['video_id']}. Skipping: {e}"
                )
                failure = torch.tensor(True, device=accelerator.device)
                continue
            for video_id, frame_idx, encoded_frames in zip(
                batch["video_id"],
                batch["frame_idx"],
                batch_encoded_frames.split(
                    [len(frame_idx) for frame_idx in batch["frame_idx"]]
                ),
                strict=True,
            ):
                # NOTE: CRITICAL REGION! Acquire a lock!
                lock = FileLock(f"{video_id}.lock")
                # we combine the previously encoded frames with the new ones
                # and save the slices to save disk space.
                encoded_frames = encoded_frames.to(
                    torch.device("cpu"), getattr(torch, torch_dtype)
                )
                frames_file = results_dir / f"{video_id}.pt"
                if frames_file.exists():
                    finished_frames_dict = torch.load(frames_file)
                    finished_frame_idx = list(finished_frames_dict.keys())
                    frame_idx.extend(finished_frame_idx)
                    encoded_frames = torch.cat(
                        [encoded_frames]
                        + [
                            finished_frames_dict[frame_id].unsqueeze(0)
                            for frame_id in finished_frame_idx
                        ]
                    )
                encoded_frame_dict = {
                    frame_id: encoded_frame
                    for frame_id, encoded_frame in zip(
                        frame_idx,
                        encoded_frames,
                        strict=True,
                    )
                }
                frames_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(encoded_frame_dict, frames_file)
                # NOTE: END OF CRITICAL REGION. Release the lock.
                lock.release()
                progress_queue.put(1)
    success = (~torch.any(accelerator.gather(failure))).item()
    # signal the progress bar process to exit
    if accelerator.is_main_process:
        progress_queue.put(None)
    accelerator.end_training()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(auto_cli(run, as_positional=False))
