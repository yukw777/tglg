import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import wandb
from jsonargparse import auto_cli
from tqdm import tqdm

from real_time_vlm_benchmark.eval.real_time_gen_eval import (
    compute_f1_score,
    compute_final_score,
)


def get_video_to_time_to_action(
    original_annotation_dir: Path,
) -> dict[str, dict[float, str]]:
    video_to_time_to_action = {}
    for label_file in Path(original_annotation_dir).glob("**/Labels-v2.json"):
        time_to_action_1 = {}
        time_to_action_2 = {}
        with open(label_file) as f:
            anns = json.load(f)
            for ann in anns["annotations"]:
                time = float(ann["position"]) / 1000
                if ann["gameTime"].startswith("1"):
                    time_to_action_1[time] = ann["label"]
                elif ann["gameTime"].startswith("2"):
                    time_to_action_2[time] = ann["label"]

        video_name = label_file.parent.name
        video_to_time_to_action[f"{video_name}/1_224p"] = time_to_action_1
        video_to_time_to_action[f"{video_name}/2_224p"] = time_to_action_2

    return video_to_time_to_action


def get_matching_actions(data: dict, time_to_actions: dict[float, str]) -> list[str]:
    # get all actions within 5 seconds of start and end times
    if "gen_start" in data and "gen_end" in data:
        start_time = float(data["gen_start"]) - 5
        end_time = float(data["gen_end"]) + 5
    else:
        start_time = float(data["start"]) - 5
        if "end" in data:
            end_time = float(data["end"]) + 5
        else:
            # assume 150 wpm for speech
            if "content" in data:
                content = data["content"]
            else:
                raise ValueError(f"No content: {data}")
            end_time = float(data["start"]) + len(content.split()) / (150 / 60) + 5
    return [
        action
        for time, action in time_to_actions.items()
        if start_time <= time <= end_time
    ]


def get_results_by_action(
    entity: str,
    project: str,
    run_name: str,
    video_to_time_to_action: dict[str, dict[float, str]],
    table_key: str,
) -> dict[str, list]:
    wandb_api = wandb.Api()
    runs = wandb_api.runs(f"{entity}/{project}", filters={"displayName": run_name})
    assert len(runs) == 1
    run = runs[0]
    table = run.logged_artifacts()[0][table_key]
    df = table.get_dataframe()
    results_by_action = defaultdict(list)
    for _, row in tqdm(
        df.iterrows(), desc=f"Get {table_key} results by action", total=len(df)
    ):
        if row["video_id"] not in video_to_time_to_action:
            # some games don't have action annotations so skip
            continue
        matching_actions = get_matching_actions(
            row, video_to_time_to_action[row["video_id"]]
        )
        for action in matching_actions:
            results_by_action[action].append({"action": action, **row})

    return results_by_action


def main(
    original_annotation_dir: Path,
    inference_entity: str,
    inference_project: str,
    inference_run_name: str,
    eval_entity: str,
    eval_project: str,
    eval_run_name: str,
    ground_truth_file: Path,
    out_file: Path,
) -> None:
    video_to_time_to_action = get_video_to_time_to_action(original_annotation_dir)
    generated_by_action = get_results_by_action(
        inference_entity,
        inference_project,
        inference_run_name,
        video_to_time_to_action,
        "inference",
    )
    eval_results_by_action = get_results_by_action(
        eval_entity, eval_project, eval_run_name, video_to_time_to_action, "eval"
    )

    ground_truth_by_action: dict[str, list] = defaultdict(list)
    with open(ground_truth_file, newline="") as f:
        ground_truth = json.load(f)
    for video_id, utterances in tqdm(ground_truth.items(), desc="Read ground truth"):
        if video_id not in video_to_time_to_action:
            # some games don't have action annotations so skip
            continue
        for utter in utterances:
            matching_actions = get_matching_actions(
                utter, video_to_time_to_action[video_id]
            )
            for action in matching_actions:
                ground_truth_by_action[action].append({"action": action, **utter})

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "action",
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

        for action, eval_results in eval_results_by_action.items():
            f1_score = compute_f1_score(
                eval_results,
                generated_by_action[action],
                ground_truth_by_action[action],
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
                    "action": action,
                    "mean_start_score": mean_start_score,
                    "mean_stop_score": mean_stop_score,
                    "mean_overlap_score": mean_overlap_score,
                    **final_score,
                    **f1_score,
                }
            )


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
