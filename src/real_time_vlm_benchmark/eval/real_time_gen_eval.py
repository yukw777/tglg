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
    tau_time: float = 3.0,
    time_window: float = 5.0,
    max_swap_passes: int = 5,
) -> list[tuple[int, int]]:
    """
    Three-Stage alignment of generated and ground-truth utterances.

    In Stage 1, we perform time-based bi-partite matching. In Stage 2, we perform
    local refinement by semantic similarity in onder to account for the fact that
    some utterances may occur out of order within a local time window. Specifically,

    STAGE 1 (Time-based bi-partite matching):
      - cost_matrix = time_penalty only (no semantic similarity).
      - row_ind, col_ind = linear_sum_assignment(...) => alignment_1

    STAGE 2 (Local refinement by semantic similarity):
      - For each matched pair (g_i, gt_j), check if there's a potential
        local swap with another pair (g_i', gt_j') in the same time_window
        that yields better overall semantic similarity.
      - Perform these swaps if beneficial.
      - Repeat up to `max_swap_passes` times to avoid infinite loops.

    STAGE 3: Final filtering => if a matched pair (g_i, gt_j) has time_diff > time_window,
      we drop it from alignment.

    Returns:
      list of (g_idx, gt_idx) pairs after refinement.
    """
    num_gen, num_gt = len(generated), len(ground_truth)
    # -------------------------
    # STAGE 1: TIME-BASED BI-PARTITE MATCHING
    # -------------------------
    # construct cost matrix using only time_penalty = exp(-time_diff / tau_time)
    cost_matrix = np.full((num_gen, num_gt), -np.inf)

    for i, gen in enumerate(generated):
        for j, gt in enumerate(ground_truth):
            time_diff = abs(gen["start"] - gt["start"])
            time_penalty = np.exp(-time_diff / tau_time)
            cost_matrix[i, j] = time_penalty

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    alignment = {g_idx: gt_idx for g_idx, gt_idx in zip(row_ind, col_ind)}

    # -------------------------
    # STAGE 2: LOCAL REFINEMENT BY SEMANTIC SIMILARITY
    # -------------------------
    # we perform up to `max_swap_passes` passes where we try local improvements
    # by swapping pairs if it increases the total local similarity.
    def is_within_time_window(g_idx, gt_idx):
        """Check if generated utterance i and GT utterance j are within time_window."""
        return (
            abs(generated[g_idx]["start"] - ground_truth[gt_idx]["start"])
            <= time_window
        )

    improved = True
    passes = 0
    while improved and passes < max_swap_passes:
        improved = False
        passes += 1

        # iterate over each matched pair
        for g_i, gt_j in list(alignment.items()):
            # The current semantic similarity
            # current_sim = local_semantic_score(g_i, gt_j)

            # look for possible local swaps among neighbors
            # i.e. other gen utterances g_i' that are matched to gt_j'
            # such that BOTH g_i, g_i' are within time_window to BOTH gt_j, gt_j'
            for g_i_prime, gt_j_prime in list(alignment.items()):
                if g_i_prime == g_i:
                    continue  # same pair => skip

                # Check if (g_i, gt_j_prime) and (g_i_prime, gt_j) are within the time_window
                if is_within_time_window(g_i, gt_j_prime) and is_within_time_window(
                    g_i_prime, gt_j
                ):
                    # Evaluate if swapping leads to better combined similarity
                    sim_original = (
                        similarity_matrix[g_i, gt_j]
                        + similarity_matrix[g_i_prime, gt_j_prime]
                    )
                    sim_swapped = (
                        similarity_matrix[g_i, gt_j_prime]
                        + similarity_matrix[g_i_prime, gt_j]
                    )

                    # If swapped sum is better => do swap
                    if sim_swapped > sim_original:
                        # Swap the matches
                        alignment[g_i] = gt_j_prime
                        alignment[g_i_prime] = gt_j
                        improved = True

    # -------------------------
    # STAGE 3: FINAL FILTER => DROP IMPROBABLE MATCHES
    # -------------------------
    final_matches = []
    for g_i, gt_j in alignment.items():
        time_diff = abs(generated[g_i]["start"] - ground_truth[gt_j]["start"])
        if time_diff <= time_window:
            final_matches.append((g_i, gt_j))
        # else: we skip => unmatched

    return final_matches


def compute_accuracy_scores(
    matched_pairs: list[tuple[int, int]],
    generated: list[dict],
    ground_truth: list[dict],
    similarity_matrix: np.ndarray,
) -> np.ndarray:
    acc_scores = []
    for gen_id, gt_id in matched_pairs:
        # cosine similarity is [-1, 1], but we want [0, 1] (accuracy)
        acc_scores.append((similarity_matrix[gen_id][gt_id] + 1) / 2)
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
    mean_acc = float(accuracy_scores.mean()) * f1_score
    mean_timing = float(timing_scores.mean()) * f1_score
    return {
        "mean_acc": mean_acc,
        "mean_timing": mean_timing,
        "final_score": alpha * mean_acc + (1 - alpha) * mean_timing,
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
