import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import transformers
from videollm_online.models import LiveTrainingArguments, get_args_class

from real_time_vlm_benchmark.baseline_models.utils.train import (
    DataCollatorForVideoLLMOnline,
    TrainArguments,
    init_model_tokenizer_for_training,
    train_preprocess,
)
from real_time_vlm_benchmark.baseline_models.videollm_online_models.soccernet import (
    soccernet_system_message,
)
from real_time_vlm_benchmark.datasets import SoccerNetDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class DataArguments:
    soccernet_dir: str
    video_frame_dir: str


def train() -> None:
    # parse args
    (args, _, _) = transformers.HfArgumentParser(
        # We have to pass DataArguments and TrainArguments here so HFArgumentParser doesn't
        # complain about unused arguments
        (LiveTrainingArguments, DataArguments, TrainArguments)
    ).parse_args_into_dataclasses()
    videollm_online_args, data_args, train_args = transformers.HfArgumentParser(
        (get_args_class(args.live_version), DataArguments, TrainArguments)
    ).parse_args_into_dataclasses()

    # initialize model and tokenizer
    model, tokenizer = init_model_tokenizer_for_training(
        videollm_online_args, train_args
    )

    # set up data processing
    video_stats: dict | None = None
    if train_args.video_stats_file is not None:
        with open(train_args.video_stats_file) as f:
            video_stats = json.load(f)
    preprocessor = partial(
        train_preprocess,
        max_num_frames=videollm_online_args.max_num_frames,
        frame_resolution=[model.config.frame_resolution] * 2,
        use_encoded_frames=not train_args.set_vision_inside,
        tokenizer=tokenizer,
        sys_msg_fn=soccernet_system_message,
        frame_token_interval_id=model.config.frame_token_interval_id,
        v_placeholder_id=model.config.v_placeholder_id,
        stream_generation_prompt_ids=tokenizer.apply_chat_template(
            [{}], add_stream_generation_prompt=True, return_tensors="pt"
        ).squeeze(0)
        if train_args.videollm_online_variant == "default"
        else tokenizer(
            "\nAssistant:", return_tensors="pt", add_special_tokens=False
        ).input_ids.squeeze(0),
        eos_token_id=model.config.eos_token_id,
        frame_num_tokens=model.config.frame_num_tokens,
        sample_fps=videollm_online_args.frame_fps,
        videollm_online_variant=train_args.videollm_online_variant,
        video_stats=video_stats,
    )
    train_dataset = SoccerNetDataset(
        Path(data_args.soccernet_dir) / "real-time-eval-annotation_train.json",
        video_dir_path=data_args.soccernet_dir,
        video_frame_dir_path=data_args.video_frame_dir,
        preprocessor=preprocessor,
    )
    val_dataset = SoccerNetDataset(
        Path(data_args.soccernet_dir) / "real-time-eval-annotation_val.json",
        video_dir_path=data_args.soccernet_dir,
        video_frame_dir_path=data_args.video_frame_dir,
        preprocessor=preprocessor,
    )

    # Track the best model
    videollm_online_args.metric_for_best_model = "eval_loss"
    # Workaround for https://github.com/huggingface/transformers/issues/26969
    # and https://github.com/huggingface/transformers/issues/23018
    videollm_online_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    # This is to avoid an extra autograph traversal every iteration, since we know
    # there are no unused parameters.
    videollm_online_args.ddp_find_unused_parameters = False
    trainer = transformers.Trainer(
        model=model,
        processing_class=tokenizer,
        args=videollm_online_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForVideoLLMOnline(tokenizer),
    )
    trainer.train(resume_from_checkpoint=videollm_online_args.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    train()
