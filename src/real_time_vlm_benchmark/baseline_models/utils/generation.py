import math
from itertools import zip_longest
from typing import Callable, TypedDict

import torch
from transformers import PreTrainedTokenizerBase

from real_time_vlm_benchmark.datasets.utils import chunked

GenerationConfig = TypedDict(
    "GenerationConfig",
    {
        "max_new_tokens": int,
        "do_sample": bool,
        "num_beams": int,
        "temperature": float,
        "top_k": int,
        "top_p": float,
    },
    total=False,
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


def tokenize_real_time_interleaved_dialogue(
    tokenizer: PreTrainedTokenizerBase,
    v_placeholder_id: int,
    eos_token_id: int,
    frame_num_tokens: int,
    sample_fps: int,
    num_total_frames: int,
    num_interleaved_frames: int,
    interleaved_dialogue: list[dict],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    def handle_text_utterance(
        tokens: list[int], labels: list[int], text_utter: dict, num_frames: int
    ) -> int:
        """
        Interleave frame tokens and text tokens and return the number of remaining frame tokens.
        """
        role_prefix = "\nUser:" if text_utter["role"] == "user" else "\nAssistant:"
        role_tokens = tokenizer(role_prefix, add_special_tokens=False).input_ids
        tokens.extend(role_tokens)
        if text_utter["role"] == "user":
            labels.extend([-100] * len(role_tokens))
        else:
            # causal shift
            labels.pop()
            labels.extend(role_tokens)

        tokenized_content = tokenizer(
            f" {text_utter['content']}", add_special_tokens=False
        ).input_ids
        if num_frames == 0:
            # no frames left, so just append
            tokens.extend(tokenized_content)
            if text_utter["role"] == "user":
                labels.extend([-100] * len(tokenized_content))
            else:
                labels.extend(tokenized_content)
                labels.append(eos_token_id)
            return num_frames

        # interleave overlapping frame tokens and text tokens
        num_overlapped_frames = min(
            math.ceil((text_utter["end"] - text_utter["start"]) * sample_fps),
            num_frames,
        )
        frame_tokens = [v_placeholder_id] * num_overlapped_frames
        if num_overlapped_frames > len(tokenized_content):
            # when there are more frames, we want to keep the chunk size small,
            # so we end this utterance with frames.
            chunk_size = num_overlapped_frames // len(tokenized_content)
            chunked_frame_tokens = list(chunked(frame_tokens, chunk_size))
            chunked_tokenized_content = [[text] for text in tokenized_content]
        else:
            # when there are more text tokens, we want to push the chunk size to be big,
            # so we end this utterance with frames.
            chunk_size = math.ceil(len(tokenized_content) / num_overlapped_frames)
            chunked_frame_tokens = [[frame] for frame in frame_tokens]
            chunked_tokenized_content = list(chunked(tokenized_content, chunk_size))
        for i, (text, frames) in enumerate(
            zip_longest(chunked_tokenized_content, chunked_frame_tokens)
        ):
            # text tokens always come first
            if text is not None:
                tokens.extend(text)
                if text_utter["role"] == "user":
                    labels.extend([-100] * len(text))
                else:
                    labels.extend(text)
                    if i == len(chunked_tokenized_content) - 1:
                        # if last set of text tokens, append eos token for causal shifting
                        labels.append(eos_token_id)
            if frames is not None:
                tokens.extend(frames * frame_num_tokens)
                labels.extend([-100] * len(frames) * frame_num_tokens)

        return num_frames - num_overlapped_frames

    tokens: list[int] = []
    labels: list[int] = []
    curr_text_utter: dict | None = None
    for i, utter in enumerate(interleaved_dialogue):
        if utter["role"] == "system":
            assert i == 0 and len(tokens) == 0
            text_tokens = tokenizer(
                f"{utter['content']}\n", add_special_tokens=False
            ).input_ids
            tokens.extend(text_tokens)
            labels.extend([-100] * len(text_tokens))
        elif utter["role"] == "stream":
            if curr_text_utter is None:
                # no corresponding text utterance so just add
                num_v_placeholders = frame_num_tokens * utter["num_frames"]
                tokens.extend([v_placeholder_id] * num_v_placeholders)
                labels.extend([-100] * num_v_placeholders)
            else:
                remainder = handle_text_utterance(
                    tokens, labels, curr_text_utter, utter["num_frames"]
                )

                # add the rest of the frame tokens, if any
                num_v_placeholders = frame_num_tokens * remainder
                tokens.extend([v_placeholder_id] * num_v_placeholders)
                labels.extend([-100] * num_v_placeholders)

                # reset curr_text_utter
                curr_text_utter = None
        else:
            if curr_text_utter is not None:
                assert f"A textual utterance without a corresponding stream utterance: {utter}"
            assert utter["role"] in {"assistant", "user"}
            curr_text_utter = utter
    if curr_text_utter is not None:
        # we have a straggler text utterance without a corresponding stream utterance
        # so take some frames from the remaining frames
        # NOTE: we don't want to too many extra frames here, otherwise we end with a
        # frame token instead of a text token, e.g., eos. So we calculate the tokens per
        # frame value, then divide the number of tokens of the straggler text utterance.
        # We're tokenizing it twice, but this is cleaner.
        tokenized_content = tokenizer(
            f" {curr_text_utter['content']}", add_special_tokens=False
        ).input_ids
        # Assume 150 wpm and 1.3 tokens per word
        num_tokens_per_frame = math.ceil(150 * 1.3 / 60 / sample_fps)
        num_extra_frames = min(
            math.ceil(len(tokenized_content) / num_tokens_per_frame),
            num_total_frames - num_interleaved_frames,
        )
        remainder = handle_text_utterance(
            tokens, labels, curr_text_utter, num_extra_frames
        )
        num_interleaved_frames += num_extra_frames - remainder

    # remove the trailing frame tokens
    num_trailing_frame_tokens = 0
    for token in reversed(tokens):
        if token == v_placeholder_id:
            num_trailing_frame_tokens += 1
        else:
            break
    if num_trailing_frame_tokens > 0:
        tokens = tokens[:-num_trailing_frame_tokens]
        labels = labels[:-num_trailing_frame_tokens]
    return (
        torch.tensor(tokens),
        torch.tensor(labels),
        num_interleaved_frames - num_trailing_frame_tokens // frame_num_tokens,
    )
