import os

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object, set_seed, tqdm
from jsonargparse import auto_cli
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset

from real_time_vlm_benchmark.baseline_models.videollm_online_models.holo_assist import (
    VideoLLMOnlineHoloAssistModel,
)
from real_time_vlm_benchmark.datasets.holo_assist import HoloAssistDataset

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(
    dataset: HoloAssistDataset,
    model: VideoLLMOnlineHoloAssistModel | None = None,
    per_device_batch_size: int = 1,
    num_dataloader_workers: int = 4,
    start_idx: int | None = None,
    end_idx: int | None = None,
    gen_config: dict | None = None,
    wandb_project: str | None = None,
    random_seed: int = 42,
    out_file_name: str | None = None,
) -> None:
    set_seed(random_seed)
    if model is None:
        model = VideoLLMOnlineHoloAssistModel()

    if gen_config is None:
        gen_config = {}
    if wandb_project is not None:
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(wandb_project)
    else:
        accelerator = Accelerator()

    model.eval()
    dataset.preprocessor = model.preprocess
    eval_dataset: Dataset = dataset
    if start_idx is not None or end_idx is not None:
        eval_dataset = Subset(dataset, range(start_idx or 0, end_idx or len(dataset)))
    model, dataloader = accelerator.prepare(
        model,
        DataLoader(
            eval_dataset,
            batch_size=per_device_batch_size,
            num_workers=num_dataloader_workers,
            pin_memory=True,
            collate_fn=model.collate_fn,
        ),
    )

    model = model.module if isinstance(model, DistributedDataParallel) else model
    results_table = pd.DataFrame(columns=["video", "start", "content"])
    for batch in tqdm(dataloader, desc="Inference"):
        with torch.inference_mode():
            preds = model.predict(batch, **gen_config)
        gathered_preds = gather_object(preds)
        if (
            accelerator.gradient_state.end_of_dataloader
            and accelerator.gradient_state.remainder > 0
        ):
            # we have some duplicates, so filter them out
            # this logic is from gather_for_metrics()
            gathered_preds = gathered_preds[: accelerator.gradient_state.remainder]
        for video, utters in gathered_preds:
            rows = {
                "video": [video] * len(utters),
                "start": [u["start"] for u in utters],
                "content": [u["content"] for u in utters],
            }
            # pd.concat raises an annoying warning when some dataframes are empty,
            # so let's just filter it out
            if results_table.empty:
                results_table = pd.DataFrame(rows)
            else:
                results_table = pd.concat([results_table, pd.DataFrame(rows)])

    if accelerator.is_main_process and out_file_name is not None:
        results_table.to_csv(out_file_name, index=False)

    if wandb_project is not None:
        accelerator.get_tracker("wandb").log_table("inference", dataframe=results_table)

    accelerator.end_training()


if __name__ == "__main__":
    auto_cli(run, as_positional=False)
