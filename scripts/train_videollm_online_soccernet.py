import os
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import transformers
from peft import PeftModel
from videollm_online.models import LiveTrainingArguments, get_args_class
from videollm_online.models.live_llama import LiveLlamaConfig, LiveLlamaForCausalLM
from videollm_online.models.tokenization_live import (
    build_live_tokenizer_and_update_config,
)

from real_time_vlm_benchmark.baseline_models.utils.train import (
    DataCollatorForVideoLLMOnline,
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


@dataclass
class TrainArguments:
    pretrained_videollm_online: str
    set_vision_inside: bool = False


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
    model = PeftModel.from_pretrained(
        model, train_args.pretrained_videollm_online, is_trainable=True
    )
    if train_args.set_vision_inside:
        model.set_vision_inside()

    # set up data processing
    preprocessor = partial(
        train_preprocess,
        frame_fps=videollm_online_args.frame_fps,
        max_num_frames=videollm_online_args.max_num_frames,
        frame_resolution=[model.config.frame_resolution] * 2,
        use_encoded_frames=not train_args.set_vision_inside,
        tokenizer=tokenizer,
        sys_msg_fn=soccernet_system_message,
        frame_token_interval_id=model.config.frame_token_interval_id,
        v_placeholder_id=model.config.v_placeholder_id,
        stream_generation_prompt_ids=tokenizer.apply_chat_template(
            [{}], add_stream_generation_prompt=True, return_tensors="pt"
        ).squeeze(0),
        eos_token_id=model.config.eos_token_id,
    )
    train_dataset = SoccerNetDataset(
        data_args.soccernet_dir,
        str(Path(data_args.soccernet_dir) / "real-time-eval-annotation_train.json"),
        video_frame_dir_path=data_args.video_frame_dir,
        preprocessor=preprocessor,
    )
    val_dataset = SoccerNetDataset(
        data_args.soccernet_dir,
        str(Path(data_args.soccernet_dir) / "real-time-eval-annotation_val.json"),
        video_frame_dir_path=data_args.video_frame_dir,
        preprocessor=preprocessor,
    )

    # Load the best model at the end so we can save it
    videollm_online_args.load_best_model_at_end = True
    # Workaround for https://github.com/huggingface/transformers/issues/26969
    videollm_online_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    # Workaround for https://github.com/huggingface/transformers/issues/23018
    videollm_online_args.ddp_find_unused_parameters = False
    trainer = transformers.Trainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForVideoLLMOnline(tokenizer),
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    train()
