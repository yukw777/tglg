import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import wandb
from jsonargparse import auto_cli

from real_time_vlm_benchmark.eval.real_time_gen_eval import (
    compute_f1_score,
    compute_final_score,
)


def get_video_to_task(original_annotation_file: Path) -> dict[str, str]:
    with open(original_annotation_file) as f:
        anns = json.load(f)
    return {ann["video_name"]: ann["taskType"] for ann in anns}


def get_results_by_task(
    entity: str,
    project: str,
    run_name: str,
    video_to_task: dict[str, str],
    table_key: str,
) -> dict[str, list]:
    wandb_api = wandb.Api()
    runs = wandb_api.runs(f"{entity}/{project}", filters={"displayName": run_name})
    assert len(runs) == 1
    run = runs[0]
    table = run.logged_artifacts()[0][table_key]
    df = table.get_dataframe()
    results_by_task = defaultdict(list)
    for _, row in df.iterrows():
        task_type = video_to_task[row["video_id"]]
        results_by_task[task_type].append({"task_type": task_type, **row})

    return results_by_task


def main(
    original_annotation_file: Path,
    inference_entity: str,
    inference_project: str,
    inference_run_name: str,
    eval_entity: str,
    eval_project: str,
    eval_run_name: str,
    ground_truth_file: Path,
    out_file: Path,
) -> None:
    video_to_task = get_video_to_task(original_annotation_file)
    generated_by_task = get_results_by_task(
        inference_entity,
        inference_project,
        inference_run_name,
        video_to_task,
        "inference",
    )
    eval_results_by_task = get_results_by_task(
        eval_entity, eval_project, eval_run_name, video_to_task, "eval"
    )

    ground_truth_by_task: dict[str, list] = defaultdict(list)
    with open(ground_truth_file, newline="") as f:
        ground_truth = json.load(f)
    for video_id, utterances in ground_truth.items():
        task_type = video_to_task[video_id]
        ground_truth_by_task[task_type].extend(
            utter for utter in utterances if utter["eval"]
        )

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_type",
                "final_score",
                "mean_acc_f1_adjusted",
                "mean_acc",
                "mean_timing_f1_adjusted",
                "mean_timing",
                "mean_start_score",
                "mean_stop_score",
                "mean_overlap_score",
                "f1",
                "prec",
                "recall",
            ],
        )
        writer.writeheader()

        for task_type, eval_results in eval_results_by_task.items():
            f1_score = compute_f1_score(
                eval_results,
                generated_by_task[task_type],
                ground_truth_by_task[task_type],
            )
            mean_start_score = float(
                np.array([result["start"] for result in eval_results]).mean()
            )
            mean_stop_score = float(
                np.array([result["stop"] for result in eval_results]).mean()
            )
            mean_overlap_score = float(
                np.array([result["overlap"] for result in eval_results]).mean()
            )
            final_score = compute_final_score(
                np.array([result["accuracy"] for result in eval_results]),
                np.array([result["timing"] for result in eval_results]),
                f1_score["f1"],
            )
            writer.writerow(
                {
                    "task_type": task_type,
                    "mean_start_score": mean_start_score,
                    "mean_stop_score": mean_stop_score,
                    "mean_overlap_score": mean_overlap_score,
                    **final_score,
                    **f1_score,
                }
            )


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
