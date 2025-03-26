from dataclasses import asdict
from typing import Callable

import cv2
import torch
from transformers import OffloadedCache

# decord must be imported after torch
# https://github.com/dmlc/decord/issues/293
import decord  # isort: skip
from decord import VideoReader
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.v2.functional import resize
from tqdm import tqdm
from videollm_online.models import build_model_and_tokenizer
from videollm_online.models.arguments_live import get_args_class

from real_time_vlm_benchmark.baseline_models import BaselineModel
from real_time_vlm_benchmark.baseline_models.utils.sample import (
    sample_frames_for_dialogue,
)


def construct_interleaved_dialogue(
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
        elif not utterance["eval"]:
            i = curr_frame_count
            while frame_timestamps[i] <= utterance["start"]:
                i += 1
            num_frames = i - curr_frame_count
            if num_frames == 0:
                # no associated video frames for this utterance, so skip
                continue
            interleaved_dialogue.append(
                {
                    "role": "stream",
                    "num_frames": i - curr_frame_count,
                    "learn": False,
                }
            )
            curr_frame_count = i
            interleaved_dialogue.append(utterance)
        else:
            break
    return interleaved_dialogue, curr_frame_count


class VideoLLMOnlineModel(BaselineModel):
    def __init__(
        self,
        version: str = "live1+",
        checkpoint: str = "chenjoya/videollm-online-8b-v1plus",
        frame_token_interval_threshold: float = 0.725,
        show_progress: bool = False,
        set_vision_inside: bool = False,
        sys_msg_fn: Callable[[list[dict]], str] | None = None,
    ) -> None:
        super().__init__()
        args = get_args_class(version)(
            resume_from_checkpoint=checkpoint,
            # live1's frame_token_interval is set to '', which doesn't quite work, so let's override it.
            # See https://github.com/showlab/videollm-online/issues/32#issuecomment-2346180840 for more details.
            frame_token_interval=",",
        )
        self.show_progress = show_progress
        self.set_vision_inside = set_vision_inside
        if sys_msg_fn is not None:
            self.sys_msg_fn = sys_msg_fn
        else:

            def default_sys_msg_fn(dialogue: list[dict]) -> str:
                return args.system_prompt

            self.sys_msg_fn = default_sys_msg_fn
        self.model, self.tokenizer = build_model_and_tokenizer(
            is_training=False, set_vision_inside=self.set_vision_inside, **asdict(args)
        )
        self.frame_num_tokens = self.model.config.frame_num_tokens
        self.frame_fps = args.frame_fps
        self.max_num_frames = args.max_num_frames
        # the separator token that separates frame tokens, i.e., ","
        self.frame_token_interval_id = self.model.config.frame_token_interval_id
        # Copied from https://github.com/showlab/videollm-online/blob/755e2652a651ae2e43bb2f9d28281a2540a3bbac/demo/inference.py#L31
        self.frame_token_interval_threshold = frame_token_interval_threshold
        self.register_buffer(
            "stream_generation_prompt_ids",
            self.tokenizer.apply_chat_template(
                [{}], add_stream_generation_prompt=True, return_tensors="pt"
            ),
            persistent=False,
        )
        self.stream_generation_prompt_ids: torch.Tensor  # For type checkers

    def preprocess(self, datapoint: dict) -> dict[str, torch.Tensor]:
        dialogue = datapoint["dialogue"]

        vidcap = cv2.VideoCapture(str(datapoint["video_path"]))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_idx, start_time, _ = sample_frames_for_dialogue(
            dialogue,
            fps,
            self.frame_fps,
            max_num_frames=self.max_num_frames,
        )
        frame_timestamps = frame_idx / fps
        if self.set_vision_inside:
            decord.bridge.set_bridge("torch")
            vr = VideoReader(str(datapoint["video_path"]))
            frames = vr.get_batch(frame_idx)
            frames = rearrange(frames, "t h w c -> t c h w")
            frames = resize(frames, [self.model.config.frame_resolution] * 2)
        else:
            encoded_frame_dict = torch.load(datapoint["encoded_frames_path"])
            frames = torch.stack(
                [encoded_frame_dict[frame_id] for frame_id in frame_idx.tolist()]
            )

        # construct an interleaved dialogue
        dialogue = [
            utter
            for utter in dialogue
            if utter.get("start", float("inf")) >= start_time
        ]
        interleaved_dialogue, num_interleaved_frames = construct_interleaved_dialogue(
            dialogue, frame_timestamps.tolist(), sys_msg_fn=self.sys_msg_fn
        )

        # tokenize the interleave dialogue
        input_ids = self.tokenizer.apply_chat_template(
            interleaved_dialogue, return_tensors="pt", add_stream_prompt=True
        ).squeeze(0)

        # divide frames into context_frames to be included in the context
        # and eval_frames that will be "streamed" for evaluation.
        context_frames = frames[:num_interleaved_frames]
        eval_frames = frames[num_interleaved_frames:]

        return {
            "index": datapoint["index"],
            "input_ids": input_ids,
            "context_frames": context_frames,
            "eval_frames": eval_frames,
            "frame_timestamps": frame_timestamps,
            "video_id": datapoint["video_id"],
        }

    @property
    def collate_fn(self) -> Callable[[list[dict]], dict]:
        def collate(datapoints: list[dict]) -> dict:
            context_frames = torch.cat([dp["context_frames"] for dp in datapoints])
            eval_frames = torch.cat([dp["eval_frames"] for dp in datapoints])
            padded = self.tokenizer.pad(
                [{"input_ids": dp["input_ids"]} for dp in datapoints]
            )
            frame_timestamps = pad_sequence(
                [dp["frame_timestamps"] for dp in datapoints], batch_first=True
            )
            video_id = [dp["video_id"] for dp in datapoints]
            idx = torch.tensor([dp["index"] for dp in datapoints])
            return {
                "index": idx,
                "context_frames": context_frames,
                "eval_frames": eval_frames,
                "frame_timestamps": frame_timestamps,
                "input_ids": padded.input_ids,
                "attention_mask": padded.attention_mask,
                "video_id": video_id,
            }

        return collate

    @torch.inference_mode()
    def predict(
        self, batch: dict, use_offloaded_cache: bool = False, **gen_kwargs
    ) -> dict[int, list]:
        # NOTE: we don't support batch prediction
        assert batch["input_ids"].size(0) == 1, "Batch prediction not supported"

        # process context_frames and input_ids
        # NOTE: we maintain attention_mask despite not supporting batch prediction,
        # because the model doesn't have a pad token set, so attention_mask can't be inferred.
        attention_mask = batch["attention_mask"]
        joint_embeds = self.model.joint_embed(
            batch["input_ids"], batch["context_frames"]
        )
        if use_offloaded_cache:
            outputs = self.model(
                inputs_embeds=joint_embeds, attention_mask=attention_mask
            )
        else:
            outputs = self.model(
                inputs_embeds=self.model.joint_embed(
                    batch["input_ids"], batch["context_frames"]
                ),
                attention_mask=attention_mask,
                past_key_values=OffloadedCache(),
            )

        index = batch["index"][0].item()
        results: dict[int, list] = {index: []}
        eval_frames = batch["eval_frames"]
        for i in tqdm(
            range(eval_frames.size(0)), desc="Frames", disable=not self.show_progress
        ):
            # keep generating one token at a time until it's ready to generate,
            # i.e., the next token is not frame_token_interval
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        attention_mask.size(0),
                        self.frame_num_tokens,
                        device=self.model.device,
                    ),
                ],
                dim=1,
            )
            outputs = self.model(
                inputs_embeds=self.model.visual_embed(eval_frames[i : i + 1]).unsqueeze(
                    0
                ),
                past_key_values=outputs.past_key_values,
                attention_mask=attention_mask,
            )
            next_token_probs = outputs.logits[:, -1:].softmax(dim=-1)
            # if the probability of frame_token_interval is below the threshold,
            # zero it out. This is b/c the model is biased towards not responding.
            if (
                next_token_probs[:, :, self.frame_token_interval_id]
                < self.frame_token_interval_threshold
            ):
                next_token_probs[:, :, self.frame_token_interval_id].zero_()
            next_token_id = next_token_probs.argmax(dim=-1)
            if next_token_id == self.frame_token_interval_id:
                # we don't generate, just append frame_token_interval and keep going
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token_id)], dim=1
                )
                outputs = self.model(
                    input_ids=next_token_id,
                    past_key_values=outputs.past_key_values,
                    attention_mask=attention_mask,
                )
            else:
                # NOTE: the original code has it, but when using live1 (with live1+ weights),
                # this assert is tripped, so let's disable it for now.
                # https://github.com/showlab/videollm-online/blob/755e2652a651ae2e43bb2f9d28281a2540a3bbac/demo/inference.py#L44
                # assert next_token_id == 933, (
                #     f"{next_token_id} != 933 (]\\n)"
                # )  # 933 = ]\n
                # we generate with the stream generation prompt
                input_ids = torch.cat(
                    [
                        # NOTE: we have to prepend these 1's due to the way generation with cache works.
                        # See https://github.com/huggingface/transformers/issues/36151 for more details.
                        torch.ones_like(attention_mask, dtype=torch.int64),
                        self.stream_generation_prompt_ids,
                    ],
                    dim=1,
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones_like(self.stream_generation_prompt_ids),
                    ],
                    dim=1,
                )
                outputs = self.model.generate(
                    input_ids=input_ids,
                    past_key_values=outputs.past_key_values,
                    attention_mask=attention_mask,
                    return_dict_in_generate=True,
                    # Set to suppress warning
                    pad_token_id=self.tokenizer.pad_token_id,
                    **gen_kwargs,
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            attention_mask.size(0),
                            # one fewer for the eos token
                            outputs.sequences.size(1) - input_ids.size(1) - 1,
                            device=self.model.device,
                        ),
                    ],
                    dim=1,
                )
                response = self.tokenizer.batch_decode(
                    outputs.sequences[:, input_ids.size(1) :], skip_special_tokens=True
                )[0].strip()
                results[index].append(
                    {
                        "video_id": batch["video_id"][0],
                        "role": "assistant",
                        "content": response,
                        "start": batch["frame_timestamps"][
                            :, batch["context_frames"].size(0) + i
                        ].item(),
                    }
                )

        return results
