import math
from dataclasses import asdict
from typing import Callable

import torch

# decord must be imported after torch
# https://github.com/dmlc/decord/issues/293
import decord  # isort: skip
from decord import VideoReader
from einops import rearrange
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.v2.functional import resize
from tqdm import tqdm
from videollm_online.models import build_model_and_tokenizer
from videollm_online.models.arguments_live import get_args_class


def sample_frames_for_dialogue(
    start_time: float, end_time: float, video_avg_fps: float, sample_fps: float
) -> torch.Tensor:
    """
    Returns the indices for frames sampled at the given fps from [start_time, end_time].
    """
    start_time_frame = math.ceil(start_time * video_avg_fps)
    end_time_frame = math.floor(end_time * video_avg_fps)
    num_frames = end_time_frame - start_time_frame + 1
    frame_interval = video_avg_fps / sample_fps
    num_frames_to_sample = math.ceil(num_frames / frame_interval)
    return torch.linspace(start_time_frame, end_time_frame, num_frames_to_sample).to(
        torch.int
    )


def construct_interleaved_dialogue(
    dialogue: list[dict], sample_fps: float, use_narration: bool = False
) -> tuple[list[dict], int]:
    """
    Construct an interleaved dialogue with frames and utterances, and return the number of frames taken
    for the interleaved dialogue
    """
    sys_msg = (
        (
            "A multimodal AI assistant is helping users with some activities. "
            "Below is their conversation, interleaved with the list of video frames received by the assistant. "
            "The assistant should give the user instructions and correct their mistakes."
        )
        if use_narration
        else "A multimodal AI assistant is helping users with some activities. Below is their conversation, interleaved with the list of video frames received by the assistant."
    )
    interleaved_dialogue: list[dict] = [{"role": "system", "content": sys_msg}]
    last_frame_idx = 0
    for utterance in dialogue:
        if utterance["role"] == "system":
            if use_narration:
                interleaved_dialogue[0]["content"] += (
                    f" Here's the summary of the activity: {utterance['content']}"
                )
            else:
                continue
        elif not utterance["eval"]:
            # add frames [last_frame_idx, frame(utterance['start'])]
            utter_start_frame_idx = math.floor(utterance["start"] * sample_fps)
            # time_interval = utterance["start"] - last_frame_time
            time_interval_num_frames = utter_start_frame_idx + 1 - last_frame_idx
            interleaved_dialogue.append(
                {
                    "role": "stream",
                    "num_frames": time_interval_num_frames,
                    "learn": False,
                }
            )
            last_frame_idx += time_interval_num_frames
            interleaved_dialogue.append(utterance)
        else:
            break
    return interleaved_dialogue, last_frame_idx


class VideoLLMOnlineHoloAssistModel(nn.Module):
    def __init__(
        self,
        version: str = "live1+",
        checkpoint: str = "chenjoya/videollm-online-8b-v1plus",
        frame_token_interval_threshold: float = 0.725,
        use_narration: bool = False,
        show_progress: bool = True,
        set_vision_inside: bool = False,
    ) -> None:
        super().__init__()
        args = get_args_class(version)(
            resume_from_checkpoint=checkpoint,
            # live1's frame_token_interval is set to '', which doesn't quite work, so let's override it.
            # See https://github.com/showlab/videollm-online/issues/32#issuecomment-2346180840 for more details.
            frame_token_interval=",",
        )
        self.use_narration = use_narration
        self.show_progress = show_progress
        self.set_vision_inside = set_vision_inside
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
        decord.bridge.set_bridge("torch")
        vr = VideoReader(str(datapoint["video"]))
        dialogue = datapoint["dialogue"]

        # sample frames from max(0, end time - max_num_frames/frame_fps) to the end time of the last utterance at self.frame_fps
        end_time = dialogue[-1]["end"]
        start_time = max(0, end_time - self.max_num_frames / self.frame_fps)
        frame_idx = sample_frames_for_dialogue(
            start_time, end_time, vr.get_avg_fps(), self.frame_fps
        )
        frame_timestamps = frame_idx / vr.get_avg_fps()
        if self.set_vision_inside:
            frames = vr.get_batch(frame_idx)
            frames = rearrange(frames, "t h w c -> t c h w")
            frames = resize(frames, [self.model.config.frame_resolution] * 2)
        else:
            frames = torch.load(datapoint["video_frame"])

        # construct an interleaved dialogue
        dialogue = [
            utter
            for utter in dialogue
            if utter.get("start", float("inf")) >= start_time
        ]
        interleaved_dialogue, num_interleaved_frames = construct_interleaved_dialogue(
            dialogue, self.frame_fps, use_narration=self.use_narration
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
            "video": datapoint["video"].parent.parent.name,
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
            video = [dp["video"] for dp in datapoints]
            idx = torch.tensor([dp["index"] for dp in datapoints])
            return {
                "index": idx,
                "context_frames": context_frames,
                "eval_frames": eval_frames,
                "frame_timestamps": frame_timestamps,
                "input_ids": padded.input_ids,
                "attention_mask": padded.attention_mask,
                "video": video,
            }

        return collate

    @torch.inference_mode()
    def predict(self, batch: dict, **gen_kwargs) -> dict[int, list]:
        # NOTE: we don't support batch prediction
        assert batch["input_ids"].size(0) == 1, "Batch prediction not supported"

        # process context_frames and input_ids
        # NOTE: we maintain attention_mask despite not supporting batch prediction,
        # because the model doesn't have a pad token set, so attention_mask can't be inferred.
        attention_mask = batch["attention_mask"]
        outputs = self.model(
            inputs_embeds=self.model.joint_embed(
                batch["input_ids"], batch["context_frames"]
            ),
            attention_mask=attention_mask,
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
                        "video": batch["video"][0],
                        "role": "assistant",
                        "content": response,
                        "start": batch["frame_timestamps"][
                            :, batch["context_frames"].size(0) + i
                        ].item(),
                    }
                )

        return results
