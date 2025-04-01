import math
from typing import Callable, TypedDict

import torch
from transformers import PreTrainedTokenizerBase

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
    frame_num_tokens: int,
    sample_fps: int,
    num_total_frames: int,
    num_interleaved_frames: int,
    interleaved_dialogue: list[dict],
) -> tuple[torch.Tensor, int]:
    def handle_text_utterance(
        tokens: list[int], text_utter: dict, num_frames: int
    ) -> int:
        """
        Interleave frame tokens and text tokens and return the number of remaining frame tokens.
        """
        role_prefix = "\nUser:" if text_utter["role"] == "user" else "\nAssistant:"
        tokens.extend(tokenizer(role_prefix, add_special_tokens=False).input_ids)

        # interleave overlapping frame tokens and text tokens
        num_overlapped_frames = min(
            math.ceil((text_utter["end"] - text_utter["start"])) * sample_fps,
            num_frames,
        )
        frame_tokens = [
            [v_placeholder_id] * frame_num_tokens for _ in range(num_overlapped_frames)
        ]
        tokenized_content = tokenizer(
            f" {text_utter['content']}", add_special_tokens=False
        ).input_ids
        if num_overlapped_frames > len(tokenized_content):
            longer_seq = frame_tokens
            interval = math.ceil(num_overlapped_frames / len(tokenized_content))
        else:
            longer_seq = tokenized_content
            interval = math.ceil(len(tokenized_content) / num_overlapped_frames)
        j = 0
        while j < len(longer_seq):
            # frame tokens always come first
            if frame_tokens == longer_seq:
                tokens.extend(
                    v
                    for v_placeholder_id_seq in frame_tokens[
                        j * interval : (j + 1) * interval
                    ]
                    for v in v_placeholder_id_seq
                )
                tokens.extend(tokenized_content[j : j + 1])
            else:
                tokens.extend(
                    v
                    for v_placeholder_id_seq in frame_tokens[j : j + 1]
                    for v in v_placeholder_id_seq
                )
                tokens.extend(tokenized_content[j * interval : (j + 1) * interval])
            j += 1
        # add the remainder from the longer sequence, if any
        if longer_seq == frame_tokens:
            tokens.extend(
                v
                for v_placeholder_id_seq in frame_tokens[j * interval :]
                for v in v_placeholder_id_seq
            )
        else:
            tokens.extend(tokenized_content[j * interval :])

        return num_frames - num_overlapped_frames

    tokens: list[int] = []
    curr_text_utter: dict | None = None
    for i, utter in enumerate(interleaved_dialogue):
        if utter["role"] == "system":
            assert i == 0 and len(tokens) == 0
            tokens.extend(
                tokenizer(
                    f"{tokenizer.bos_token}{utter['content']}\n",
                    add_special_tokens=False,
                ).input_ids
            )
        elif utter["role"] == "stream":
            if curr_text_utter is None:
                # no corresponding text utterance so just add
                tokens.extend(
                    [v_placeholder_id] * frame_num_tokens * utter["num_frames"]
                )
            else:
                remainder = handle_text_utterance(
                    tokens, curr_text_utter, utter["num_frames"]
                )

                # add the rest of the frame tokens, if any
                tokens.extend([v_placeholder_id] * frame_num_tokens * remainder)

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
            f" {curr_text_utter['content']}{tokenizer.eos_token if curr_text_utter['role'] == 'assistant' else ''}",
            add_special_tokens=False,
        ).input_ids
        # Assume 150 wpm and 1.3 tokens per word
        num_tokens_per_frame = math.ceil(150 * 1.3 / 60 / sample_fps)
        num_extra_frames = min(
            len(tokenized_content) // num_tokens_per_frame,
            num_total_frames - num_interleaved_frames,
        )
        if num_extra_frames > 0:
            remainder = handle_text_utterance(tokens, curr_text_utter, num_extra_frames)
            num_interleaved_frames += num_extra_frames - remainder

    return torch.tensor(tokens), num_interleaved_frames
