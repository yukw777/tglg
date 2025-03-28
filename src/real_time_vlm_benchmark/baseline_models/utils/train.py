import torch

# decord must be imported after torch
# https://github.com/dmlc/decord/issues/293
import decord  # isort: skip
from typing import Callable

from decord import VideoReader
from einops import rearrange
from torchvision.transforms.v2.functional import resize
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

from real_time_vlm_benchmark.baseline_models.utils.generation import (
    tokenize_real_time_interleaved_dialogue,
)
from real_time_vlm_benchmark.baseline_models.utils.sample import (
    sample_frames_for_dialogue,
)


def construct_interleaved_dialogue_for_training(
    dialogue: list[dict],
    frame_timestamps: list[float],
    sys_msg_fn: Callable[[list[dict]], str],
) -> tuple[list[dict], int]:
    """
    Construct an interleaved dialogue with frames and utterances, and return the number of frames taken
    for the interleaved dialogue
    """
    interleaved_dialogue: list[dict] = [
        {"role": "system", "content": sys_msg_fn(dialogue)}
    ]
    curr_frame_count = 0
    for utterance in dialogue:
        if utterance["role"] == "system":
            # we assume sys_msg_fn() took care of this
            continue
        i = curr_frame_count
        while i < len(frame_timestamps) and frame_timestamps[i] <= utterance["start"]:
            i += 1
        num_frames = i - curr_frame_count
        if num_frames == 0:
            # no associated video frames for this utterance, so skip
            continue
        interleaved_dialogue.append({"role": "stream", "num_frames": num_frames})
        curr_frame_count = i
        interleaved_dialogue.append(utterance)
    return interleaved_dialogue, curr_frame_count


def generate_labels(
    input_ids: torch.Tensor,
    frame_token_interval_id: int,
    v_placeholder_id: int,
    stream_generation_prompt_ids: torch.Tensor,
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.full_like(input_ids, -100)
    i = 0
    while i < input_ids.size(0):
        # if "," (frame token interval), check if they're between <v>'s (v_placeholder)
        if input_ids[i] == frame_token_interval_id:
            if (
                i > 0
                and input_ids[i - 1] == v_placeholder_id
                and i < input_ids.size(0)
                and input_ids[i + 1] == v_placeholder_id
            ):
                labels[i] = input_ids[i]

        if (
            # if stream_generation_prompt ("]\n", "Assistant", ":"), we use everything up to the next "<|eot_id|>", inclusive.
            i + 3 < input_ids.size(0)
            and input_ids[i : i + 3].equal(stream_generation_prompt_ids)
        ) or (
            # if stream_generation_prompt ("Assistant", ":"), we use everything up to the next "<|eot_id|>", inclusive.
            i + 2 < input_ids.size(0)
            and input_ids[i : i + 2].equal(stream_generation_prompt_ids[1:])
        ):
            j = i
            while j < input_ids.size(0) and input_ids[j] != eos_token_id:
                labels[j] = input_ids[j]
                j += 1
            if j < input_ids.size(0) and input_ids[j] == eos_token_id:
                labels[j] = input_ids[j]
            i = j

        i += 1

    # causal shift labels and input_ids
    return input_ids[:-1], labels[1:]


def generate_real_time_labels(
    input_ids: torch.Tensor,
    v_placeholder_id: int,
    stream_generation_prompt_ids: torch.Tensor,
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.full_like(input_ids, -100)
    i = 0
    while i < input_ids.size(0):
        if (
            # if stream_generation_prompt ("\n", "Assistant", ":"),
            # we use everything except for "<v>"'s up to the next "<|eot_id|>", inclusive.
            i + 3 < input_ids.size(0)
            and input_ids[i : i + 3].equal(stream_generation_prompt_ids)
        ):
            j = i
            while j < input_ids.size(0) and input_ids[j] != eos_token_id:
                if input_ids[j] != v_placeholder_id:
                    labels[j] = input_ids[j]
                j += 1
            if j < input_ids.size(0) and input_ids[j] == eos_token_id:
                labels[j] = input_ids[j]
            i = j

        i += 1

    # causal shift labels and input_ids
    return input_ids[:-1], labels[1:]


def train_preprocess(
    datapoint: dict,
    *,
    frame_fps: float,
    max_num_frames: int,
    frame_resolution: list[int],
    use_encoded_frames: bool,
    tokenizer: PreTrainedTokenizerBase,
    sys_msg_fn: Callable[[list[dict]], str],
    frame_token_interval_id: int,
    v_placeholder_id: int,
    stream_generation_prompt_ids: torch.Tensor,
    eos_token_id: int,
    frame_num_tokens: int,
    sample_fps: int,
    videollm_online_variant: str,
) -> dict[str, torch.Tensor | list[int]]:
    decord.bridge.set_bridge("torch")
    vr = VideoReader(str(datapoint["video_path"]))
    dialogue = datapoint["dialogue"]

    frame_idx, start_time, _ = sample_frames_for_dialogue(
        dialogue,
        vr.get_avg_fps(),
        frame_fps,
        len(vr),
        max_num_frames=max_num_frames,
    )
    frame_timestamps = frame_idx / vr.get_avg_fps()
    if use_encoded_frames:
        encoded_frame_dict = torch.load(datapoint["encoded_frames_path"])
        frames = torch.stack(
            [encoded_frame_dict[frame_id] for frame_id in frame_idx.tolist()]
        )
    else:
        frames = vr.get_batch(frame_idx)
        frames = rearrange(frames, "t h w c -> t c h w")
        frames = resize(frames, frame_resolution)

    # construct an interleaved dialogue
    dialogue = [
        utter for utter in dialogue if utter.get("start", float("inf")) >= start_time
    ]
    interleaved_dialogue, num_interleaved_frames = (
        construct_interleaved_dialogue_for_training(
            dialogue, frame_timestamps, sys_msg_fn=sys_msg_fn
        )
    )

    if videollm_online_variant == "default":
        # tokenize the interleave dialogue
        input_ids = tokenizer.apply_chat_template(
            interleaved_dialogue, return_tensors="pt"
        ).squeeze(0)

        # generate labels
        input_ids, labels = generate_labels(
            input_ids,
            frame_token_interval_id,
            v_placeholder_id,
            stream_generation_prompt_ids,
            eos_token_id,
        )
    elif videollm_online_variant == "real-time":
        input_ids, num_interleaved_frames = tokenize_real_time_interleaved_dialogue(
            tokenizer,
            v_placeholder_id,
            frame_num_tokens,
            sample_fps,
            frames.size(0),
            num_interleaved_frames,
            interleaved_dialogue,
        )
        input_ids, labels = generate_real_time_labels(
            input_ids, v_placeholder_id, stream_generation_prompt_ids, eos_token_id
        )
    else:
        raise ValueError(f"Unknown VideoLLM-Online variant: {videollm_online_variant}")

    return {
        "input_ids": input_ids,
        "frames": frames[:num_interleaved_frames],
        # NOTE: We return a list here to avoid having the collator converting the labels
        # into a list of numpy arrays and constructing a tensor from them, which is
        # extremely slow.
        "labels": labels.tolist(),
    }


class DataCollatorForVideoLLMOnline(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        frames = torch.cat([f.pop("frames") for f in features])
        collated = super().__call__(features, return_tensors=return_tensors)
        collated["frames"] = frames
        return collated
