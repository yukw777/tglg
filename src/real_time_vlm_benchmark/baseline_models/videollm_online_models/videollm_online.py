import json
import math
from dataclasses import asdict
from typing import Callable

import torch
from transformers import Cache, OffloadedCache
from transformers.modeling_outputs import CausalLMOutputWithPast

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
from real_time_vlm_benchmark.baseline_models.utils.generation import (
    construct_interleaved_dialogue,
    tokenize_real_time_interleaved_dialogue,
)
from real_time_vlm_benchmark.baseline_models.utils.sample import (
    sample_frames_for_dialogue,
)


class VideoLLMOnlineModel(BaselineModel):
    def __init__(
        self,
        version: str = "live1+",
        checkpoint: str = "chenjoya/videollm-online-8b-v1plus",
        frame_token_interval_threshold: float = 0.725,
        show_progress: bool = False,
        set_vision_inside: bool = False,
        sys_msg_fn: Callable[[list[dict]], str] | None = None,
        video_stats_file: str | None = None,
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

        self.video_stats: dict | None = None
        if video_stats_file is not None:
            with open(video_stats_file) as f:
                self.video_stats = json.load(f)

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

        vr: VideoReader | None = None
        if self.video_stats is None or datapoint["video_id"] not in self.video_stats:
            vr = VideoReader(str(datapoint["video_path"]))
            avg_fps = vr.get_avg_fps()
            total_num_frames = len(vr)
        else:
            stats = self.video_stats[datapoint["video_id"]]
            avg_fps = stats["fps"]
            total_num_frames = stats["num_frames"]

        frame_idx, start_time, _ = sample_frames_for_dialogue(
            dialogue,
            avg_fps,
            self.frame_fps,
            total_num_frames,
            max_num_frames=self.max_num_frames,
        )
        frame_timestamps = frame_idx / avg_fps
        if self.set_vision_inside:
            decord.bridge.set_bridge("torch")
            if (
                self.video_stats is None
                or datapoint["video_id"] not in self.video_stats
            ):
                assert vr is not None
            else:
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
            dialogue, frame_timestamps, sys_msg_fn=self.sys_msg_fn
        )

        # tokenize the interleave dialogue
        input_ids, num_interleaved_frames = self._tokenize_interleaved_dialogue(
            interleaved_dialogue, frames.size(0), num_interleaved_frames
        )

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

    def _tokenize_interleaved_dialogue(
        self,
        interleaved_dialogue: list[dict],
        num_total_frames: int,
        num_remaining_frames: int,
    ) -> tuple[torch.Tensor, int]:
        return self.tokenizer.apply_chat_template(
            interleaved_dialogue, return_tensors="pt", add_stream_prompt=True
        ).squeeze(0), num_remaining_frames

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

    def _process_context(
        self, batch: dict, use_offloaded_cache: bool
    ) -> CausalLMOutputWithPast:
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
                inputs_embeds=self.model.joint_embed(
                    batch["input_ids"], batch["context_frames"]
                ),
                attention_mask=attention_mask,
                past_key_values=OffloadedCache(),
            )
        else:
            outputs = self.model(
                inputs_embeds=joint_embeds, attention_mask=attention_mask
            )
        return outputs

    @torch.inference_mode()
    def predict(
        self, batch: dict, use_offloaded_cache: bool = False, **gen_kwargs
    ) -> dict[int, list]:
        outputs = self._process_context(batch, use_offloaded_cache)

        index = batch["index"][0].item()
        results: dict[int, list] = {index: []}
        eval_frames = batch["eval_frames"]
        attention_mask = batch["attention_mask"]
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


class RealTimeModel(VideoLLMOnlineModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer(
            "stream_generation_prompt_ids",
            self.tokenizer(
                "\nAssistant:", return_tensors="pt", add_special_tokens=False
            ).input_ids,
            persistent=False,
        )
        # Assume 150 wpm and 1.3 tokens per word
        # for frame_fps of 2, the number of tokens per frame is 2
        self.num_tokens_per_frame = math.ceil(150 * 1.3 / 60 / self.frame_fps)
        # we stop generation on eos_token or \n (the start of the stream generation prompt)
        self.eos_token_id_list = [
            self.tokenizer.eos_token_id,
            self.tokenizer.encode("\n", add_special_tokens=False)[0],
        ]

    def _tokenize_interleaved_dialogue(
        self,
        interleaved_dialogue: list[dict],
        num_total_frames: int,
        num_remaining_frames: int,
    ) -> tuple[torch.Tensor, int]:
        input_ids, _, num_interleaved_frames = tokenize_real_time_interleaved_dialogue(
            self.tokenizer,
            self.model.config.v_placeholder_id,
            self.tokenizer.eos_token_id,
            self.frame_num_tokens,
            self.frame_fps,
            num_total_frames,
            num_remaining_frames,
            interleaved_dialogue,
        )
        return input_ids, num_interleaved_frames

    def _generate(
        self,
        attention_mask: torch.Tensor,
        past_key_values: Cache,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **gen_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, Cache]:
        assert not (input_ids is not None and inputs_embeds is not None), (
            "Only one of input_ids and inputs_embeds can be set"
        )
        args = {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "return_dict_in_generate": True,
            # we generate up to num_tokens_per_frame
            "max_new_tokens": self.num_tokens_per_frame,
            # Set to suppress warning
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.eos_token_id_list,
        }
        if input_ids is not None:
            outputs = self.model.generate(input_ids=input_ids, **args, **gen_kwargs)
        else:
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds, **args, **gen_kwargs
            )
        if input_ids is not None:
            generated_input_ids = outputs.sequences[:, input_ids.size(1) :]
        else:
            generated_input_ids = outputs.sequences
        if generated_input_ids[0, -1] not in self.eos_token_id_list:
            # process the last token so that it's cached
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        attention_mask.size(0),
                        generated_input_ids.size(1),
                        device=self.model.device,
                    ),
                ],
                dim=1,
            )
            outputs = self.model(
                input_ids=generated_input_ids[:, -1:],
                past_key_values=outputs.past_key_values,
                attention_mask=attention_mask,
            )
        else:
            # no need to process, just update the attention_mask without the eos token
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        attention_mask.size(0),
                        generated_input_ids.size(1) - 1,
                        device=self.model.device,
                    ),
                ],
                dim=1,
            )

        return generated_input_ids, attention_mask, outputs.past_key_values

    @torch.inference_mode()
    def predict(
        self, batch: dict, use_offloaded_cache: bool = False, **gen_kwargs
    ) -> dict[int, list]:
        # we manage max_new_tokens manually
        max_new_tokens = gen_kwargs.pop("max_new_tokens", None)

        outputs = self._process_context(batch, use_offloaded_cache)
        past_key_values = outputs.past_key_values

        index = batch["index"][0].item()
        results: dict[int, list] = {index: []}
        eval_frames = batch["eval_frames"]
        # NOTE: we maintain attention_mask despite not supporting batch prediction,
        # because the model doesn't have a pad token set, so attention_mask can't be inferred.
        attention_mask = batch["attention_mask"]
        curr_utter: dict | None = None
        for i in tqdm(
            range(eval_frames.size(0)), desc="Frames", disable=not self.show_progress
        ):
            # NOTE: this is an important invariant, so we check it every iteration to be safe.
            assert attention_mask.size(1) == past_key_values.get_seq_length()

            # encode the current frame
            inputs_embeds = self.model.visual_embed(eval_frames[i : i + 1]).unsqueeze(0)
            # update attention_mask for the current frame
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
            if curr_utter is None:
                # we haven't started generating, so see if the next token is "\n" (198)
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    past_key_values=outputs.past_key_values,
                    attention_mask=attention_mask,
                )
                past_key_values = outputs.past_key_values
                next_token_probs = outputs.logits[:, -1:].softmax(dim=-1)
                next_token_id = next_token_probs.argmax(dim=-1)
                if next_token_id == self.stream_generation_prompt_ids[:, 0]:
                    # feed the model the stream generation prompt to start generating
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
                    generated_input_ids, attention_mask, past_key_values = (
                        self._generate(
                            attention_mask,
                            past_key_values,
                            input_ids=input_ids,
                            **gen_kwargs,
                        )
                    )
                    response = self.tokenizer.decode(
                        generated_input_ids[0],
                        skip_special_tokens=True,
                    )
                    curr_utter = {
                        "video_id": batch["video_id"][0],
                        "role": "assistant",
                        "content": response,
                        "start": batch["frame_timestamps"][
                            :, batch["context_frames"].size(0) + i
                        ].item(),
                    }
            else:
                # we're already generating so keep generating and appending
                inputs_embeds = torch.cat(
                    [
                        # we have to pad inputs_embeds due to the way
                        # hugging face handles generating with inputs_embeds
                        # and KV cache. The padding is automatically stripped.
                        torch.zeros(
                            inputs_embeds.size(0),
                            past_key_values.get_seq_length(),
                            inputs_embeds.size(2),
                            dtype=inputs_embeds.dtype,
                            device=inputs_embeds.device,
                        ),
                        inputs_embeds,
                    ],
                    dim=1,
                )
                generated_input_ids, attention_mask, past_key_values = self._generate(
                    attention_mask,
                    past_key_values,
                    inputs_embeds=inputs_embeds,
                    **gen_kwargs,
                )
                response = self.tokenizer.decode(
                    generated_input_ids[0],
                    skip_special_tokens=True,
                )
                curr_utter["content"] += response
                if (
                    len(curr_utter["content"]) > max_new_tokens
                    or generated_input_ids[0, -1] in self.eos_token_id_list
                ):
                    # done with this one, so append and start a new utterance
                    curr_utter["content"] = curr_utter["content"].strip()
                    results[index].append(curr_utter)
                    curr_utter = None

        return results
