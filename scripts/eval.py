import json
import re
from collections import defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
import wandb
from jsonargparse import auto_cli
from sentence_transformers import SentenceTransformer

from real_time_vlm_benchmark.eval.real_time_gen_eval import (
    align_utterances,
    compute_accuracy_scores,
    compute_f1_score,
    compute_final_score,
    compute_similarity_matrix,
    compute_timing_scores,
)


def eval(
    model: SentenceTransformer,
    generated: dict[str, list[dict]],
    ground_truth: dict[str, list[dict]],
) -> dict:
    individual_eval_results = []
    total_matched_pairs = []
    total_generated = []
    total_ground_truth = []
    total_accuracy_scores = np.array([])
    total_timing_scores = np.array([])
    total_start_scores = np.array([])
    total_stop_scores = np.array([])
    total_overlap_scores = np.array([])
    for video, gt_utters in ground_truth.items():
        total_ground_truth.extend(gt_utters)
        total_generated.extend(generated[video])

        sim_mat = compute_similarity_matrix(model, generated[video], gt_utters)
        matched_pairs = align_utterances(generated[video], gt_utters, sim_mat)
        total_matched_pairs.extend(matched_pairs)

        accuracy_scores = compute_accuracy_scores(
            matched_pairs, generated[video], gt_utters, sim_mat
        )
        total_accuracy_scores = np.concat([total_accuracy_scores, accuracy_scores])
        timing_scores = compute_timing_scores(
            matched_pairs, generated[video], gt_utters
        )
        total_timing_scores = np.concat(
            [total_timing_scores, timing_scores["timing_scores"]]
        )
        total_start_scores = np.concat(
            [total_start_scores, timing_scores["start_scores"]]
        )
        total_stop_scores = np.concat([total_stop_scores, timing_scores["stop_scores"]])
        total_overlap_scores = np.concat(
            [total_overlap_scores, timing_scores["overlap_scores"]]
        )

        for acc, timing, start, stop, overlap, (matched_gen, matched_gt) in zip(
            accuracy_scores,
            timing_scores["timing_scores"],
            timing_scores["start_scores"],
            timing_scores["stop_scores"],
            timing_scores["overlap_scores"],
            matched_pairs,
            strict=True,
        ):
            individual_eval_results.append(
                {
                    "video": video,
                    "accuracy": float(acc),
                    "timing": float(timing),
                    "start": float(start),
                    "stop": float(stop),
                    "overlap": float(overlap),
                    "gen_start": generated[video][matched_gen]["start"],
                    "gt_start": gt_utters[matched_gt]["start"],
                    "gt_end": gt_utters[matched_gt]["end"],
                    "gen_content": generated[video][matched_gen]["content"],
                    "gt_content": gt_utters[matched_gt]["content"],
                }
            )
    f1_score = compute_f1_score(
        total_matched_pairs, total_generated, total_ground_truth
    )
    final_score = compute_final_score(
        total_accuracy_scores, total_timing_scores, f1_score["f1"]
    )
    return {
        "individual_eval_results": individual_eval_results,
        **final_score,
        **f1_score,
    }


def read_ground_truth(ground_truth_file: str) -> dict[str, list[dict]]:
    with open(ground_truth_file) as f:
        raw_gt = json.load(f)
    preprocessed = {}
    for video, utterances in raw_gt.items():
        preprocessed[video] = [utter for utter in utterances if utter["eval"]]

    return preprocessed


def preprocess_inference_results(results: pd.DataFrame) -> dict[str, list[dict]]:
    preprocessed = defaultdict(list)
    for _, row in results.iterrows():
        preprocessed[row["video"]].append(
            {"content": row["content"], "start": float(row["start"])}
        )
    return preprocessed


def main(
    ground_truth_file: str,
    inference_entity: str,
    inference_project: str,
    inference_run_name_regex: str,
    eval_project: str,
    eval_entity: str | None = None,
    sent_sim_model_name: str = "all-mpnet-base-v2",
) -> None:
    sent_sim_model = SentenceTransformer(sent_sim_model_name)
    ground_truth = read_ground_truth(ground_truth_file)

    if eval_entity is None:
        eval_entity = inference_entity
    wandb_api = wandb.Api()
    pattern = re.compile(inference_run_name_regex)
    inference_runs = [
        run
        for run in wandb_api.runs(f"{inference_entity}/{inference_project}")
        if pattern.search(run.name)
    ]
    print("Running evaluation for the following inference runs:")
    pprint([run.name for run in inference_runs])
    print("===========================================")
    for inference_run in inference_runs:
        table = inference_run.logged_artifacts()[0]["inference"]
        df = table.get_dataframe()
        generated = preprocess_inference_results(df)
        eval_results = eval(sent_sim_model, generated, ground_truth)
        results_table = pd.DataFrame(eval_results.pop("individual_eval_results"))
        print(f"==== Metrics for {inference_run.name} ====")
        pprint(eval_results)
        print("===========================================")
        with wandb.init(
            entity=eval_entity, project=eval_project, name=inference_run.name
        ) as eval_run:
            eval_results["eval"] = wandb.Table(dataframe=results_table)
            eval_run.log(eval_results)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
