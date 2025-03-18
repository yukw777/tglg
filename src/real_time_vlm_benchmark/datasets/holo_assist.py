import json
from pathlib import Path
from typing import Any, Callable

from torch.utils.data import Dataset

from real_time_vlm_benchmark.datasets.utils import convert_real_time_anns_to_datapoint


def convert_holo_assist(
    holo_assist_anns: list[dict],
) -> dict[str, list[dict[str, Any]]]:
    anns: dict[str, list[dict[str, Any]]] = {}
    for holo_assist_ann in holo_assist_anns:
        dialogue: list[dict[str, Any]] = []
        correction_found = False
        last_correction_end = -1

        convs = [
            event
            for event in holo_assist_ann["events"]
            if event["label"] == "Conversation"
            # filter out low confidence transcriptions
            and (
                "Transcription Confidence" not in event["attributes"]
                or event["attributes"]["Transcription Confidence"] == "none"
            )
        ]
        for i, conv in enumerate(convs):
            utter = {
                "role": "assistant"
                if conv["attributes"]["Conversation Purpose"].startswith("instructor")
                else "user",
                "content": conv["attributes"]["Transcription"],
                "start": conv["start"],
                "end": conv["end"],
            }
            # if within 10 seconds of the last correction, we include for the evaluation
            if last_correction_end == -1 or utter["start"] >= last_correction_end + 10:
                utter["eval"] = False
            else:
                utter["eval"] = True

            # look for a correction
            if conv["attributes"]["Conversation Purpose"].endswith(
                "correct the wrong action"
            ):
                correction_found = True
                last_correction_end = conv["end"]
                utter["eval"] = True
                # look back and mark eval = True until the first `instruction`
                j = i - 1
                while j >= 0:
                    if dialogue[j]["role"] == "assistant":
                        dialogue[j]["eval"] = True
                    if convs[j]["attributes"]["Conversation Purpose"].endswith(
                        "instruction"
                    ):
                        break
                    j -= 1
            dialogue.append(utter)

        if correction_found:
            # Add the narration as a system message
            # Using this is optional for models.
            for event in holo_assist_ann["events"]:
                if event["label"] == "Narration":
                    dialogue = [
                        {
                            "role": "system",
                            "content": event["attributes"]["Long form description"],
                            "eval": False,
                        }
                    ] + dialogue
            # Remove all the trailing non-eval utterances
            i = len(dialogue) - 1
            while i >= 0:
                if dialogue[i]["eval"]:
                    break
                i -= 1
            anns[holo_assist_ann["video_name"]] = dialogue[: i + 1]
    return anns


class HoloAssistDataset(Dataset):
    def __init__(
        self,
        video_dir_path: str,
        ann_file_path: str,
        video_frame_dir_path: str | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.video_dir_path = Path(video_dir_path)
        self.video_frame_dir_path = (
            Path(video_frame_dir_path) if video_frame_dir_path is not None else None
        )
        self.preprocessor = preprocessor
        with open(ann_file_path) as f:
            anns = json.load(f)
        self.data = convert_real_time_anns_to_datapoint(anns)

    def __getitem__(self, index: int) -> dict:
        video_id, dialogue = self.data[index]
        datapoint = {
            "index": index,
            "video_id": video_id,
            "video_path": self.video_dir_path
            / video_id
            / "Export_py/Video_pitchshift.mp4",
            "dialogue": dialogue,
        }
        if self.video_frame_dir_path is not None:
            datapoint["encoded_frames_dir"] = self.video_frame_dir_path / video_id
        if self.preprocessor is not None:
            return self.preprocessor(datapoint)
        return datapoint

    def __len__(self) -> int:
        return len(self.data)
