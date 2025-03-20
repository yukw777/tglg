import csv
import multiprocessing as mp
import os
import sys
from multiprocessing.managers import BaseManager
from pathlib import Path

import pandas as pd
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from jsonargparse import auto_cli
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from real_time_vlm_benchmark.baseline_models import BaselineModel
from real_time_vlm_benchmark.baseline_models.utils.generation import GenerationConfig
from real_time_vlm_benchmark.baseline_models.utils.sample import QueueSampler
from real_time_vlm_benchmark.datasets.holo_assist import HoloAssistDataset

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(
    dataset: HoloAssistDataset,
    results_dir: Path,
    model: BaselineModel,
    per_device_batch_size: int = 1,
    num_dataloader_workers: int = 4,
    start_idx: int | None = None,
    end_idx: int | None = None,
    gen_config: GenerationConfig | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    random_seed: int = 42,
    out_file_name: str | None = None,
    mp_manager_ip_addr: str = "",
    mp_manager_port: int = 12345,
    mp_manager_auth_key: bytes = b"password",
) -> int:
    set_seed(random_seed)

    # initialize accelerator
    # NOTE: accelerator has to be initialized after model initialization
    accelerator = Accelerator()

    # set up wandb
    run = None
    if accelerator.is_main_process and any(
        [
            wandb_entity is not None,
            wandb_project is not None,
            wandb_run_name is not None,
        ]
    ):
        run = wandb.init(
            entity=wandb_entity, project=wandb_project, name=wandb_run_name
        )

    # set up model
    if gen_config is None:
        gen_config = {}
    model.eval()
    model.to(accelerator.device)

    # set up the preprocessor
    dataset.preprocessor = model.preprocess
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

    filtered_dataset = Subset(dataset, sorted(inference_indices))

    # set up the queues
    class QueueManager(BaseManager):
        pass

    class ProgressBarProcess(mp.Process):
        def __init__(self, progress_queue: mp.Queue, total: int) -> None:
            self.progress_queue = progress_queue
            self.progress_bar = tqdm(total=total, desc="Inference")
            super().__init__()

        def run(self) -> None:
            while True:
                progress = self.progress_queue.get()
                if progress is None:
                    self.progress_bar.close()
                    return
                if progress >= 0:
                    self.progress_bar.update(progress)
                else:
                    self.progress_bar.total += progress
                    self.progress_bar.refresh()

    if accelerator.is_main_process:
        queue: mp.Queue[int] = mp.Queue()
        for i in range(len(filtered_dataset)):
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
        progress_bar_proc = ProgressBarProcess(progress_queue, len(filtered_dataset))
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

    dataloader = DataLoader(
        filtered_dataset,
        batch_size=per_device_batch_size,
        num_workers=num_dataloader_workers,
        pin_memory=True,
        collate_fn=model.collate_fn,
        sampler=QueueSampler(queue),
    )
    failure = torch.tensor(False, device=accelerator.device)
    for batch in dataloader:
        try:
            batch["context_frames"] = batch["context_frames"].to(accelerator.device)
            batch["eval_frames"] = batch["eval_frames"].to(accelerator.device)
            batch["input_ids"] = batch["input_ids"].to(accelerator.device)
            batch["attention_mask"] = batch["attention_mask"].to(accelerator.device)
            with torch.inference_mode():
                try:
                    preds = model.predict(batch, **gen_config)
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"[rank {accelerator.process_index}] CUDA OOM raised for batch {batch['index'].tolist()}. Retrying with OffloadedCache."
                    )
                    preds = model.predict(batch, use_offloaded_cache=True, **gen_config)
        except Exception as e:
            print(
                f"[rank {accelerator.process_index}] Exception raised for batch {batch['index'].tolist()}. Skipping: {e}"
            )
            failure = torch.tensor(True, device=accelerator.device)
            # remove this batch from the total
            progress_queue.put(-len(batch["video_id"]))
            continue
        for index, utters in preds.items():
            with open(results_dir / f"{index}.csv", "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, ["video_id", "start", "content"])
                writer.writeheader()
                for utter in utters:
                    writer.writerow(
                        {
                            "video_id": utter["video_id"],
                            "start": utter["start"],
                            "content": utter["content"],
                        }
                    )
        progress_queue.put(len(batch["video_id"]))
    success = (~torch.any(accelerator.gather(failure))).item()

    if success and accelerator.is_main_process and out_file_name is not None:
        with open(out_file_name, "w", newline="") as out_file:
            writer = csv.DictWriter(out_file, ["video_id", "start", "content"])
            writer.writeheader()
            for f in sorted(results_dir.iterdir()):
                with open(f, newline="") as in_file:
                    reader = csv.DictReader(in_file)
                    for row in reader:
                        writer.writerow(row)

    if success and accelerator.is_main_process and run is not None:
        data = []
        for f in sorted(results_dir.iterdir()):
            with open(f, newline="") as in_file:
                reader = csv.DictReader(in_file)
                for row in reader:
                    data.append(row)
        df = pd.DataFrame(data)
        table = wandb.Table(dataframe=df)
        run.log({"inference": table})

    # signal the progress bar process to exit
    if accelerator.is_main_process:
        progress_queue.put(None)
    accelerator.end_training()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(auto_cli(run, as_positional=False))
