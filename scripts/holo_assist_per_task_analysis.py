import argparse
from pathlib import Path

import pandas as pd

TASK_GROUPS = {
    "Assemble Furniture": [
        "assemble nightstand",
        "assemble stool",
        "assemble tray table",
        "assemble utility cart",
    ],
    "Disassemble Furniture": [
        "disassemble nightstand",
        "disassemble stool",
        "disassemble tray table",
        "disassemble utility cart",
    ],
    "Make Coffee": [
        "make coffee with nespresso machine",
        "make coffee with espresso machine",
    ],
    "Repair Machinery": [
        "change belt",
        "change circuit breaker",
        "fix motorcycle",
    ],
    "Setup Electronics": [
        "setup camera",
        "setup switch",
        "setup big printer",
        "setup small printer",
        "setup gopro",
        "assemble laser scanner",
        "assemble computer",
    ],
}

KEY_COL = "task_type"
SCORE_COLS = ["lin_comb_final_score", "geo_mean_final_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HoloAssist per-task TRACE scores between two CSV files."
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


def print_per_task_scores(merged: pd.DataFrame) -> None:
    print("Per-task deltas:")
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
    for group, task_types in TASK_GROUPS.items():
        subset = merged[merged[KEY_COL].isin(task_types)]
        missing_task_types = sorted(set(task_types) - set(subset[KEY_COL]))

        print(
            group,
            "n =",
            len(subset),
            "lin_comb_delta =",
            subset["delta_lin_comb"].mean(),
            "geo_mean_delta =",
            subset["delta_geo_mean"].mean(),
            "task_types =",
            subset[KEY_COL].tolist(),
        )

        if missing_task_types:
            print("  WARNING missing expected task types:", missing_task_types)


def main() -> None:
    args = parse_args()
    merged = get_merged_scores(args.vlm_tsi_file, args.videollm_online_file)
    print_per_task_scores(merged)
    print_grouped_scores(merged)


if __name__ == "__main__":
    main()
