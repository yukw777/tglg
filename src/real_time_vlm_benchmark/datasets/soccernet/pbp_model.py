import json
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import BasePredictionWriter
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split

LABELS = ["Analysis", "Play-by-Play"]
LABEL_TO_ID = {v: i for i, v in enumerate(LABELS)}


class PBPCommentaryBiLSTMClassifier(L.LightningModule):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, len(LABELS))  # *2 for bidirectional

        self.test_preds: list[int] = []
        self.test_labels: list[int] = []

    def forward(self, sent_seq: torch.Tensor) -> torch.Tensor:
        """
        sent_seq: (batch_size, sent_seq_len, input_dim)
        output: (batch_size, 2)
        """
        # h_n: (num_layers * 2, batch_size, hidden_dim)
        _, (h_n, _) = self.lstm(sent_seq)
        # Concatenate last forward and backward states
        # (batch_size, 2*hidden_dim)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        h_n = self.dropout(h_n)
        return self.fc(h_n)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        sent_seq, label = batch
        logits = self.forward(sent_seq)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        sent_seq, label = batch
        logits = self.forward(sent_seq)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, label)
        self.log("val_loss", loss)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        sent_seq, label = batch
        logits = self.forward(sent_seq)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, label)
        self.log("test_loss", loss)

        preds = torch.argmax(logits, dim=1)
        self.test_preds.extend(preds.tolist())
        self.test_labels.extend(label.tolist())

    def on_test_epoch_end(self) -> None:
        print(
            classification_report(
                self.test_labels, self.test_preds, target_names=LABELS
            )
        )

    def predict_step(self, batch: dict, batch_idx: int) -> list[dict]:
        logits = self.forward(batch.pop("sent_seq"))
        preds = logits.argmax(dim=1).tolist()

        # transpose the batch
        segments = [
            {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in zip(batch.keys(), values, strict=True)
            }
            for values in zip(*batch.values(), strict=True)
        ]
        for seg, pred in zip(segments, preds, strict=True):
            seg["is_pbp"] = pred == LABEL_TO_ID["Play-by-Play"]
        return segments

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


def construct_sent_seq(
    encoded_sents: torch.Tensor, index: int, seq_len: int
) -> torch.Tensor:
    sent = encoded_sents[index : index + 1]
    if index < seq_len:
        pad_len = seq_len - index - 1
        context = torch.cat(
            [
                torch.zeros(pad_len, encoded_sents.size(1)),
                encoded_sents[index - (seq_len - pad_len - 1) : index],
            ]
        )
    else:
        context = encoded_sents[index - (seq_len - 1) : index]
    return torch.cat([context, sent])


class PBPCommentaryDataModule(L.LightningDataModule):
    def __init__(
        self,
        seq_len: int = 15,
        train_val_test_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        sent_emb_model_name: str = "all-mpnet-base-v2",
        seed: int = 42,
        train_batch_size: int = 32,
        val_batch_size: int = 128,
        test_batch_size: int = 128,
        predict_batch_size: int = 128,
        labeled_transcript_dir: str | None = None,
        predict_transcript_file: str | None = None,
    ) -> None:
        super().__init__()
        self.labeled_transcript_dir = (
            Path(labeled_transcript_dir) if labeled_transcript_dir is not None else None
        )
        self.predict_transcript_file = predict_transcript_file
        self.seq_len = seq_len
        assert sum(train_val_test_ratio) == 1
        self.train_val_test_ratio = train_val_test_ratio
        self.sent_emb_model_name = sent_emb_model_name
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.predict_batch_size = predict_batch_size

    def setup(self, stage: str) -> None:
        if stage in ("fit", "test"):
            assert self.labeled_transcript_dir is not None
            # read the data
            segments = []
            for labeled_transcript in self.labeled_transcript_dir.glob("**/*.json"):
                with open(labeled_transcript) as f:
                    data = json.load(f)
                segments.extend(data["segments"])

            # encode sentences
            sent_emb_model = SentenceTransformer(self.sent_emb_model_name)
            sents = [seg["text"] for seg in segments]
            encoded_sents = torch.from_numpy(
                sent_emb_model.encode(sents, show_progress_bar=True)
            )

            # construct sentence sequences and their labels
            sent_seq = torch.stack(
                [
                    construct_sent_seq(encoded_sents, i, self.seq_len)
                    for i in range(encoded_sents.size(0))
                ]
            )
            labels = torch.tensor(
                [
                    LABEL_TO_ID["Play-by-Play"]
                    if seg["is_pbp"]
                    else LABEL_TO_ID["Analysis"]
                    for seg in segments
                ]
            )

            # split train, val, test
            train_size = int(self.train_val_test_ratio[0] * len(sent_seq))
            val_size = int(self.train_val_test_ratio[1] * len(sent_seq))
            test_size = len(sent_seq) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                TensorDataset(sent_seq, labels),
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
        if stage == "predict":
            assert self.predict_transcript_file is not None
            with open(self.predict_transcript_file) as f:
                data = json.load(f)

            if len(data["segments"]) == 0:
                raise ValueError("No segments")

            # Remove words for now, and add a speaker label if it doesn't exist.
            for seg in data["segments"]:
                if "words" in seg:
                    del seg["words"]
                if "speaker" not in seg:
                    seg["speaker"] = "UNK"

            # encode sentences
            sent_emb_model = SentenceTransformer(self.sent_emb_model_name)
            sents = [seg["text"] for seg in data["segments"]]
            encoded_sents = torch.from_numpy(
                sent_emb_model.encode(sents, show_progress_bar=True)
            )

            # construct sentence sequences and their labels
            stacked_sent_seq = torch.stack(
                [
                    construct_sent_seq(encoded_sents, i, self.seq_len)
                    for i in range(encoded_sents.size(0))
                ]
            )
            self.predict_dataset = [
                {"sent_seq": sent_seq, **seg}
                for seg, sent_seq in zip(data["segments"], stacked_sent_seq)
            ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_dataset, batch_size=self.predict_batch_size)  # type: ignore


class PBPPredWriter(BasePredictionWriter):
    def __init__(self, output_file: str) -> None:
        super().__init__(write_interval="epoch")
        self.output_file = output_file

    def write_on_epoch_end(
        self, trainer, pl_module, predictions, batch_indices
    ) -> None:
        segments = [seg for preds in predictions for seg in preds]
        with open(self.output_file, "w") as f:
            json.dump({"segments": segments}, f, indent=4)
