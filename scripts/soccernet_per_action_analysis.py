import argparse
from pathlib import Path

import pandas as pd

ACTION_GROUPS = {
    "Attempts": ["Shots on target", "Shots off target", "Clearance"],
    "Discipline": ["Yellow card", "Red card", "Yellow->red card", "Yellow→red card"],
    "Goal/Penalty": ["Goal", "Penalty"],
    "Infractions": ["Offside", "Foul"],
    "Restarts": [
        "Kick-off",
        "Ball out of play",
        "Throw-in",
        "Corner",
        "Direct free-kick",
        "Indirect free-kick",
    ],
    "Substitution": ["Substitution"],
}

KEY_COL = "action"
SCORE_COLS = ["lin_comb_final_score", "geo_mean_final_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SoccerNet per-action TRACE scores between two CSV files."
    )
    parser.add_argument("vlm_tsi_file", type=Path)
    parser.add_argument("videollm_online_file", type=Path)
    return parser.parse_args()


def read_scores(file: Path) -> pd.DataFrame:
    results = pd.read_csv(file)
    cols = [KEY_COL, *SCORE_COLS]
    missing_cols = sorted(set(cols) - set(results.columns))
    if missing_cols:
        raise ValueError(f"{file} is missing columns: {missing_cols}")
    return results[cols]


def get_merged_scores(vlm_tsi_file: Path, videollm_online_file: Path) -> pd.DataFrame:
    vlm_tsi_results = read_scores(vlm_tsi_file)
    videollm_online_results = read_scores(videollm_online_file)

    merged = vlm_tsi_results.merge(
        videollm_online_results,
        on=KEY_COL,
        suffixes=("_vlm_tsi", "_videollm_online"),
        how="outer",
        indicator=True,
    )

    score_cols = [
        "lin_comb_final_score_vlm_tsi",
        "lin_comb_final_score_videollm_online",
        "geo_mean_final_score_vlm_tsi",
        "geo_mean_final_score_videollm_online",
    ]
    merged[score_cols] = merged[score_cols].fillna(0)
    merged["delta_lin_comb"] = (
        merged["lin_comb_final_score_vlm_tsi"]
        - merged["lin_comb_final_score_videollm_online"]
    )
    merged["delta_geo_mean"] = (
        merged["geo_mean_final_score_vlm_tsi"]
        - merged["geo_mean_final_score_videollm_online"]
    )
    return merged


def print_per_action_scores(merged: pd.DataFrame) -> None:
    print("Per-action deltas:")
    print(
        merged[
            [
                KEY_COL,
                "_merge",
                "lin_comb_final_score_vlm_tsi",
                "lin_comb_final_score_videollm_online",
                "delta_lin_comb",
                "geo_mean_final_score_vlm_tsi",
                "geo_mean_final_score_videollm_online",
                "delta_geo_mean",
            ]
        ]
        .sort_values(KEY_COL)
        .to_string(index=False)
    )


def print_grouped_scores(merged: pd.DataFrame) -> None:
    print("\nGrouped means:")
    for group, actions in ACTION_GROUPS.items():
        subset = merged[merged[KEY_COL].isin(actions)]
        missing_actions = sorted(set(actions) - set(subset[KEY_COL]))

        print(
            group,
            "n =",
            len(subset),
            "lin_comb_delta =",
            subset["delta_lin_comb"].mean(),
            "geo_mean_delta =",
            subset["delta_geo_mean"].mean(),
            "actions =",
            subset[KEY_COL].tolist(),
        )

        if missing_actions:
            print("  WARNING missing expected actions:", missing_actions)


def main() -> None:
    args = parse_args()
    merged = get_merged_scores(args.vlm_tsi_file, args.videollm_online_file)
    print_per_action_scores(merged)
    print_grouped_scores(merged)


if __name__ == "__main__":
    main()
