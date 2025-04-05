import torch

# decord must be imported after torch
# https://github.com/dmlc/decord/issues/293
import decord  # isort: skip
from dataclasses import asdict, dataclass
from typing import Callable

from decord import VideoReader
from einops import rearrange
from peft import PeftModel
from torchvision.transforms.v2.functional import resize
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from video_reader import PyVideoReader
from videollm_online.models import LiveTrainingArguments
from videollm_online.models.live_llama import LiveLlamaConfig, LiveLlamaForCausalLM
from videollm_online.models.tokenization_live import (
    build_live_tokenizer_and_update_config,
)

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


def train_preprocess(
    datapoint: dict,
    *,
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
    video_stats: dict | None = None,
    use_decord: bool = True,
) -> dict[str, torch.Tensor | list[int]]:
    dialogue = datapoint["dialogue"]

    vr: VideoReader | None = None
    if video_stats is None:
        vr = VideoReader(str(datapoint["video_path"]))
        avg_fps = vr.get_avg_fps()
        total_num_frames = len(vr)
    else:
        stats = video_stats[datapoint["video_id"]]
        avg_fps = stats["fps"]
        total_num_frames = stats["num_frames"]
    frame_idx, start_time, _ = sample_frames_for_dialogue(
        dialogue,
        avg_fps,
        sample_fps,
        total_num_frames,
        max_num_frames=max_num_frames,
    )
    frame_timestamps = frame_idx / avg_fps
    if use_encoded_frames:
        encoded_frame_dict = torch.load(datapoint["encoded_frames_path"])
        frames = torch.stack(
            [encoded_frame_dict[frame_id] for frame_id in frame_idx.tolist()]
        )
    else:
        if use_decord:
            decord.bridge.set_bridge("torch")
            if video_stats is None:
                assert vr is not None
            else:
                vr = VideoReader(str(datapoint["video_path"]))
            frames = vr.get_batch(frame_idx)
        else:
            # NOTE: video_reader-rs supports resizing at decoding, which helps keeping the memory usage low.
            vr = PyVideoReader(
                str(datapoint["video_path"]), resize_shorter_side=frame_resolution[0]
            )
            frames = torch.from_numpy(vr.get_batch(datapoint["frame_idx"]))
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
        input_ids, labels, num_interleaved_frames = (
            tokenize_real_time_interleaved_dialogue(
                tokenizer,
                v_placeholder_id,
                eos_token_id,
                frame_num_tokens,
                sample_fps,
                frames.size(0),
                num_interleaved_frames,
                interleaved_dialogue,
            )
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


@dataclass
class TrainArguments:
    pretrained_videollm_online: str
    videollm_online_variant: str
    set_vision_inside: bool = False
    video_stats_file: str | None = None


def init_model_tokenizer_for_training(
    videollm_online_args: LiveTrainingArguments, train_args: TrainArguments
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    # NOTE: we want to fine-tune the pretrained videollm-online,
    # so we manually load the model instead of using build_model_and_tokenizer()
    model = LiveLlamaForCausalLM.from_pretrained(
        videollm_online_args.llm_pretrained,
        config=LiveLlamaConfig.from_pretrained(
            videollm_online_args.llm_pretrained, **asdict(videollm_online_args)
        ),
        torch_dtype="auto",
        attn_implementation=videollm_online_args.attn_implementation,
    )
    tokenizer = build_live_tokenizer_and_update_config(
        videollm_online_args.llm_pretrained, model.config
    )
    # build_live_tokenizer_and_update_config() for some reason sets padding_side to left,
    # which is only necessary for inference. Let's override it.
    tokenizer.padding_side = "right"

    # HACK: PeftModel automatically infers which device the weights should be loaded,
    # and this causes during the initialization process for training as Peft tries to load
    # the weights from checkpoints or pretrained weights to the incorrect GPUs. We could
    # get around this issue by explicitly setting torch_device="cpu", but even the Hugging
    # Face Trainer itself doesn't set that internally, so we need to resort to monkeypatching.
    _original_load_adapter = PeftModel.load_adapter

    def load_adapter_to_cpu(
        self,
        model_id: str,
        adapter_name: str,
        is_trainable: bool = False,
        torch_device: str | None = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        **kwargs,
    ):
        return _original_load_adapter(
            self,
            model_id,
            adapter_name,
            is_trainable=is_trainable,
            autocast_adapter_dtype=autocast_adapter_dtype,
            ephemeral_gpu_offload=ephemeral_gpu_offload,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_device="cpu",
            **kwargs,
        )

    PeftModel.load_adapter = load_adapter_to_cpu  # type: ignore

    model = PeftModel.from_pretrained(
        model,
        train_args.pretrained_videollm_online,
        is_trainable=True,
    )
    if train_args.set_vision_inside:
        model.set_vision_inside()

    return model, tokenizer
