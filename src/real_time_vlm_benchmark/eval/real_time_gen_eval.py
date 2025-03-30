from copy import deepcopy

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from spacy import Language


def compute_similarity_matrix(
    sent_sim_model: SentenceTransformer,
    generated: list[dict],
    ground_truth: list[dict],
) -> np.ndarray:
    """
    Computes cosine similarity matrix using sentence embeddings.
    The returned similarity matrix is of shape (num_gen, num_gt).
    """
    gen_texts = [gen["content"] for gen in generated]
    gt_texts = [gt["content"] for gt in ground_truth]
    embeddings = sent_sim_model.encode(gen_texts + gt_texts, convert_to_tensor=True)
    gen_embeddings, gt_embeddings = torch.split(
        embeddings, [len(gen_texts), len(gt_texts)]
    )
    assert sent_sim_model.similarity_fn_name == "cosine"
    return sent_sim_model.similarity(gen_embeddings, gt_embeddings).cpu().numpy()


def align_utterances(
    generated: list[dict],
    ground_truth: list[dict],
    similarity_matrix: np.ndarray,
    tau_time: float = 3,
) -> list[tuple[int, int]]:
    """
    Aligns generated and ground-truth utterances using bi-partite matching.
    time_window is in seconds
    """
    num_gen, num_gt = len(generated), len(ground_truth)
    cost_matrix = np.full((num_gen, num_gt), -np.inf)

    for i, gen in enumerate(generated):
        for j, gt in enumerate(ground_truth):
            time_diff = abs(gen["start"] - gt["start"])
            time_penalty = np.exp(-time_diff / tau_time)
            cost_matrix[i, j] = similarity_matrix[i, j] * time_penalty

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    return list(zip(row_ind, col_ind, strict=True))


def compute_accuracy_scores(
    matched_pairs: list[tuple[int, int]],
    generated: list[dict],
    ground_truth: list[dict],
    similarity_matrix: np.ndarray,
) -> np.ndarray:
    acc_scores = []
    for gen_id, gt_id in matched_pairs:
        acc_scores.append(similarity_matrix[gen_id][gt_id])
    return np.array(acc_scores)


def compute_timing_scores(
    matched_pairs: list[tuple[int, int]],
    generated: list[dict],
    ground_truth: list[dict],
    tau: float = 1.0,
    score_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> dict[str, np.ndarray]:
    """Computes timing scores (start, stop, overlap) for matched utterances."""
    gen_start_list: list[float] = []
    gen_end_list: list[float] = []
    gt_start_list: list[float] = []
    gt_end_list: list[float] = []
    for gen_id, gt_id in matched_pairs:
        gen_start_list.append(generated[gen_id]["start"])
        # assume 150 wpm for speech
        gen_end_list.append(
            generated[gen_id]["start"]
            + len(generated[gen_id]["content"].split()) / (150 / 60)
        )
        gt_start_list.append(ground_truth[gt_id]["start"])
        gt_end_list.append(ground_truth[gt_id]["end"])
    gen_start = np.array(gen_start_list)
    gen_end = np.array(gen_end_list)
    gt_start = np.array(gt_start_list)
    gt_end = np.array(gt_end_list)

    start_scores = np.exp(-np.abs(gen_start - gt_start) / tau)
    stop_scores = np.exp(-np.abs(gen_end - gt_end) / tau)

    # overlap_matrix(i, j) = min(gen_end[i], gen_end[j]) - max(gen_start[i], gen_start[j])
    start_matrix = np.maximum(gen_start[:, None], gen_start)
    end_matrix = np.minimum(gen_end[:, None], gen_end)
    overlap_matrix = end_matrix - start_matrix

    # overlap(i) = sum_{i!=j} max(0, overlap_matrix(i, j))
    overlap = np.maximum(0, overlap_matrix)
    np.fill_diagonal(overlap, 0)
    overlap = np.sum(overlap, axis=1)

    overlap_scores = np.exp(-overlap / tau)
    return {
        "timing_scores": score_weights[0] * start_scores
        + score_weights[1] * stop_scores
        + score_weights[2] * overlap_scores,
        "start_scores": start_scores,
        "stop_scores": stop_scores,
        "overlap_scores": overlap_scores,
    }


def compute_f1_score(
    matched_pairs: list[tuple[int, int]],
    generated: list[dict],
    ground_truth: list[dict],
) -> dict[str, float]:
    """Computes precision, recall, and F1 score."""
    prec = len(matched_pairs) / len(generated) if len(generated) > 0 else 0
    recall = len(matched_pairs) / len(ground_truth) if len(ground_truth) > 0 else 0
    f1 = (2 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
    return {"prec": prec, "recall": recall, "f1": f1}


def compute_final_score(
    accuracy_scores: np.ndarray,
    timing_scores: np.ndarray,
    f1_score: float,
    alpha: float = 0.5,
) -> dict[str, float]:
    mean_acc = float(accuracy_scores.mean())
    mean_timing = float(timing_scores.mean())
    return {
        "mean_acc": mean_acc,
        "mean_timing": mean_timing,
        "final_score": (alpha * mean_acc + (1 - alpha) * mean_timing) * f1_score,
    }


def canonicalize(model: Language, utterances: list[dict]) -> list[dict]:
    replaced = deepcopy(utterances)

    for utter in replaced:
        doc = model(utter["content"])
        new_tokens = []
        for token in doc:
            # replace player and team names
            if token.ent_type_ in {"PERSON", "ORG"}:
                new_tokens.append(token.ent_type_ + token.whitespace_)
            else:
                new_tokens.append(token.text_with_ws)
        utter["content"] = "".join(new_tokens)

    return replaced
