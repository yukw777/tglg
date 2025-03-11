from lightning.pytorch.cli import LightningCLI

from real_time_vlm_benchmark.datasets.soccernet.pbp_model import (
    PBPCommentaryBiLSTMClassifier,
    PBPCommentaryDataModule,
    PBPPredWriter,  # noqa: F401
)


def main() -> None:
    LightningCLI(
        model_class=PBPCommentaryBiLSTMClassifier,
        datamodule_class=PBPCommentaryDataModule,
    )


if __name__ == "__main__":
    main()
