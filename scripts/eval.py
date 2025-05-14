import itertools
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Iterator

import numpy as np
import pandas as pd
import spacy
import wandb
from jsonargparse import auto_cli
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from real_time_vlm_benchmark.eval.real_time_gen_eval import (
    align_utterances,
    canonicalize,
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
    replace_proper_nouns: bool = False,
    spacy_model: spacy.Language | None = None,
) -> tuple[list[dict], dict]:
    individual_eval_results = []
    total_matched_pairs = []
    total_generated = []
    total_ground_truth = []
    total_accuracy_scores = np.array([])
    total_timing_scores = np.array([])
    total_start_scores = np.array([])
    total_stop_scores = np.array([])
    total_overlap_scores = np.array([])
    for video_id, gt_utters in tqdm(ground_truth.items(), desc="Evaluating"):
        gen_utters = generated[video_id]
        if replace_proper_nouns:
            assert spacy_model is not None
            gt_utters = canonicalize(spacy_model, gt_utters)
            gen_utters = canonicalize(spacy_model, gen_utters)
        total_ground_truth.extend(gt_utters)
        total_generated.extend(gen_utters)

        sim_mat = compute_similarity_matrix(model, gen_utters, gt_utters)
        matched_pairs = align_utterances(gen_utters, gt_utters, sim_mat)
        total_matched_pairs.extend(matched_pairs)

        accuracy_scores = compute_accuracy_scores(
            matched_pairs, gen_utters, gt_utters, sim_mat
        )
        total_accuracy_scores = np.concat([total_accuracy_scores, accuracy_scores])
        timing_scores = compute_timing_scores(matched_pairs, gen_utters, gt_utters)
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
                    "video_id": video_id,
                    "accuracy": float(acc),
                    "timing": float(timing),
                    "start": float(start),
                    "stop": float(stop),
                    "overlap": float(overlap),
                    "gen_start": gen_utters[matched_gen]["start"],
                    "gen_end": gen_utters[matched_gen]["end"],
                    "gt_start": gt_utters[matched_gt]["start"],
                    "gt_end": gt_utters[matched_gt]["end"],
                    "gen_content": gen_utters[matched_gen]["content"],
                    "gt_content": gt_utters[matched_gt]["content"],
                }
            )
    f1_score = compute_f1_score(
        total_matched_pairs, total_generated, total_ground_truth
    )
    mean_start_score = float(total_start_scores.mean())
    mean_stop_score = float(total_stop_scores.mean())
    mean_overlap_score = float(total_overlap_scores.mean())
    final_score = compute_final_score(
        total_accuracy_scores, total_timing_scores, f1_score["f1"]
    )
    return individual_eval_results, {
        "mean_start_score": mean_start_score,
        "mean_stop_score": mean_stop_score,
        "mean_overlap_score": mean_overlap_score,
        **final_score,
        **f1_score,
    }


def read_ground_truth(ground_truth_file: Path) -> dict[str, list[dict]]:
    with open(ground_truth_file) as f:
        raw_gt = json.load(f)
    preprocessed = {}
    for video_id, utterances in raw_gt.items():
        preprocessed[video_id] = [utter for utter in utterances if utter["eval"]]

    return preprocessed


def preprocess_inference_results(results: pd.DataFrame) -> dict[str, list[dict]]:
    preprocessed = defaultdict(list)
    for _, row in results.iterrows():
        # NOTE: backward compatibility
        video_id = row["video_id"] if "video_id" in row else row["video"]
        result = {"content": row["content"], "start": float(row["start"])}
        if "end" in row:
            result["end"] = float(row["end"])
        preprocessed[video_id].append(result)
    return preprocessed


@dataclass
class InferenceWandB:
    entity: str
    project: str
    run_name_regex: str


@dataclass
class InferenceLocal:
    files: list[Path]


@dataclass
class EvalWandB:
    project: str
    entity: str | None = None


@dataclass
class EvalLocal:
    files_dir: Path


def get_dataframes_wandb(
    infer_wandb: InferenceWandB,
) -> Iterator[tuple[str, pd.DataFrame]]:
    wandb_api = wandb.Api()
    pattern = re.compile(infer_wandb.run_name_regex)
    inference_runs = [
        run
        for run in wandb_api.runs(f"{infer_wandb.entity}/{infer_wandb.project}")
        if pattern.search(run.name)
    ]
    print("Running evaluation for the following inference runs:")
    pprint([run.name for run in inference_runs])
    print("===========================================")
    for inference_run in inference_runs:
        table = inference_run.logged_artifacts()[0]["inference"]
        df = table.get_dataframe()
        yield inference_run.name, df


def get_dataframes_local(
    infer_local: InferenceLocal,
) -> Iterator[tuple[str, pd.DataFrame]]:
    print("Running evaluation for the following inference files:")
    pprint([f.name for f in infer_local.files])
    print("===========================================")
    for infer_file in infer_local.files:
        yield infer_file.stem, pd.read_csv(infer_file)


def log_eval_wandb(
    name: str,
    individual_eval_results: list[dict],
    eval_results: dict,
    eval_wandb: EvalWandB,
) -> None:
    results_table = pd.DataFrame(individual_eval_results)
    with wandb.init(
        entity=eval_wandb.entity, project=eval_wandb.project, name=name
    ) as eval_run:
        eval_run.log({"eval": wandb.Table(dataframe=results_table), **eval_results})


def log_eval_local(
    name: str,
    individual_eval_results: list[dict],
    eval_results: dict,
    eval_local: EvalLocal,
) -> None:
    eval_local.files_dir.mkdir(exist_ok=True)
    results_table = pd.DataFrame(individual_eval_results)
    results_table.to_csv(eval_local.files_dir / f"{name}.csv", index=False)
    with open(eval_local.files_dir / f"{name}.json", "w") as f:
        json.dump(eval_results, f)


def main(
    ground_truth_file: Path,
    infer_wandb: InferenceWandB | None = None,
    infer_local: InferenceLocal | None = None,
    eval_wandb: EvalWandB | None = None,
    eval_local: EvalLocal | None = None,
    sent_sim_model_name: str = "all-mpnet-base-v2",
    replace_proper_nouns: bool = False,
    spacy_model_name: str = "en_core_web_lg",
) -> None:
    dfs: Iterator[tuple[str, pd.DataFrame]] = iter(())
    if infer_wandb is not None:
        dfs = itertools.chain(dfs, get_dataframes_wandb(infer_wandb))
    if infer_local is not None:
        dfs = itertools.chain(dfs, get_dataframes_local(infer_local))
    if eval_wandb is not None and eval_wandb.entity is None:
        assert infer_wandb is not None, (
            "`eval_wandb.eval_entity` is None, but infer_wandb.inference_entity is also None."
        )
        eval_wandb.entity = infer_wandb.entity
    sent_sim_model = SentenceTransformer(sent_sim_model_name)
    ground_truth = read_ground_truth(ground_truth_file)
    spacy_model = None
    if replace_proper_nouns:
        spacy_model = spacy.load(spacy_model_name)

    for name, df in dfs:
        generated = preprocess_inference_results(df)
        individual_eval_results, eval_results = eval(
            sent_sim_model,
            generated,
            ground_truth,
            replace_proper_nouns=replace_proper_nouns,
            spacy_model=spacy_model,
        )
        print(f"==== Metrics for {name} ====")
        pprint(eval_results)
        print("===========================================")
        if eval_wandb is not None:
            log_eval_wandb(name, individual_eval_results, eval_results, eval_wandb)
        if eval_local is not None:
            log_eval_local(name, individual_eval_results, eval_results, eval_local)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
