import csv
import os
from pathlib import Path

import pandas as pd
import torch
import wandb
from accelerate import PartialState
from accelerate.utils import set_seed, tqdm
from jsonargparse import auto_cli
from torch.utils.data import DataLoader, Subset

from real_time_vlm_benchmark.baseline_models.utils.generation import GenerationConfig
from real_time_vlm_benchmark.baseline_models.videollm_online_models.holo_assist import (
    VideoLLMOnlineHoloAssistModel,
)
from real_time_vlm_benchmark.datasets.holo_assist import HoloAssistDataset

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(
    dataset: HoloAssistDataset,
    results_dir: Path,
    model: VideoLLMOnlineHoloAssistModel | None = None,
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
) -> None:
    set_seed(random_seed)

    # initialize distributed_state
    distributed_state = PartialState()

    # set up wandb
    run = None
    if any(
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
    if model is None:
        model = VideoLLMOnlineHoloAssistModel()
    model.eval()
    model.to(distributed_state.device)

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
        results_dir.mkdir()

    # third, remove finished indices from inference indices
    inference_indices -= finished_idx

    # finally, take care of start_idx and end_idx
    # [start_idx, end_idx)
    if start_idx is not None:
        inference_indices = set(idx for idx in inference_indices if idx >= start_idx)
    if end_idx is not None:
        inference_indices = set(idx for idx in inference_indices if idx < end_idx)

    with distributed_state.split_between_processes(
        sorted(inference_indices)
    ) as per_process_indices:
        dataloader = DataLoader(
            Subset(dataset, per_process_indices),
            batch_size=per_device_batch_size,
            num_workers=num_dataloader_workers,
            pin_memory=True,
            collate_fn=model.collate_fn,
        )
        for batch in tqdm(dataloader, desc="Inference"):
            batch["context_frames"] = batch["context_frames"].to(
                distributed_state.device
            )
            batch["eval_frames"] = batch["eval_frames"].to(distributed_state.device)
            batch["input_ids"] = batch["input_ids"].to(distributed_state.device)
            batch["attention_mask"] = batch["attention_mask"].to(
                distributed_state.device
            )
            with torch.inference_mode():
                preds = model.predict(batch, **gen_config)
            for index, utters in preds.items():
                with open(results_dir / f"{index}.csv", "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, ["video", "start", "content"])
                    writer.writeheader()
                    for utter in utters:
                        writer.writerow(
                            {
                                "video": utter["video"],
                                "start": utter["start"],
                                "content": utter["content"],
                            }
                        )
    distributed_state.wait_for_everyone()

    if distributed_state.is_main_process and out_file_name is not None:
        with open(out_file_name, "w", newline="") as out_file:
            writer = csv.DictWriter(out_file, ["video", "start", "content"])
            writer.writeheader()
            for f in sorted(results_dir.iterdir()):
                with open(f, newline="") as in_file:
                    reader = csv.DictReader(in_file)
                    for row in reader:
                        writer.writerow(row)

    if run is not None:
        data = []
        for f in sorted(results_dir.iterdir()):
            with open(f, newline="") as in_file:
                reader = csv.DictReader(in_file)
                for row in reader:
                    data.append(row)
        df = pd.DataFrame(data)
        table = wandb.Table(dataframe=df)
        run.log({"inference": table})

    distributed_state.destroy_process_group()


if __name__ == "__main__":
    auto_cli(run, as_positional=False)
