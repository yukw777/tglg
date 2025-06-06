# Temporally-Grounded Language Generation: A Benchmark for Real-Time Vision-Language Models

Temporally-Grounded Language Generation (TGLG) is a benchmark for real-time vision-language models (VLMs) that focus on two key capabilities: perceptual updating and contingency awareness. This repository also contains code for a baseline real-time VLM for TGLG, Vision-Language Model with Time-Synchronized Interleaving (VLM-TSI).

## Download TGLG Data

First, download HoloAssist data (https://holoassist.github.io/) and SoccerNet data (https://www.soccer-net.org/) from their official websites. The TGLG annotations can be downloaded here: https://huggingface.co/datasets/kpyu/tglg.

## Released Models

- [kpyu/soccernet-videollm-online](https://huggingface.co/kpyu/soccernet-videollm-online): VideoLLM-Online fine-tuned on SoccerNet
- [kpyu/soccernet-vlm-tsi](https://huggingface.co/kpyu/soccernet-vlm-tsi): VLM-TSI fine-tuned on SoccerNet
- [kpyu/ego4d-goalstep-vlm-tsi](https://huggingface.co/kpyu/ego4d-goalstep-vlm-tsi): VLM-TSI fine-tuned on streaming free-form Ego4D GoalStep conversation data

## Installation

```bash
# If Slurm, ensure you have the latest CUDA and GCC loaded
module load cuda gcc

# Create a virtual env using uv
uv venv

# flash-attn build dependencies
uv pip install torch setuptools psutil packaging ninja

# Build flash-attn
uv pip install flash-attn --no-build-isolation

# Install the project
uv sync

# NOTE: while we recommend using uv, regular pip also works.
# pip install -e '.[flash-attn]'

# Workaround for https://github.com/OpenNMT/CTranslate2/issues/1826
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib/
```

## Development

```bash
# Create a virtual env using uv
uv venv

# flash-attn build dependencies
uv pip install torch setuptools psutil packaging ninja

# Build flash-attn
uv pip install flash-attn --no-build-isolation

# Install the project
uv sync --all-extras

# Install pre-commit hooks
pre-commit install

# Workaround for https://github.com/OpenNMT/CTranslate2/issues/1826
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib/
```

## HoloAssist TGLG Annotation Generation

Download HoloAssist data from the official website: https://holoassist.github.io/

```bash
python scripts/gen_anns_holo_assist.py \
--holo_assist_video_dir /path/to/HoloAssist/video_pitch_shifted/ \
--holo_assist_anns_file /path/to/HoloAssist/data-annotation-trainval-v1_1.json \
--output_file /path/to/HoloAssist/holo-assist-tglg-eval.json
```

## SoccerNet TGLG Annotation Generation

Download SoccerNet data from the official website: https://www.soccer-net.org/

### Transcribe Commentaries

```bash
python scripts/soccernet_transcribe.py \
--soccernet_dir path/to/SoccerNet \
--output_dir path/to/SoccerNet-transcribed/
```

### Play-by-Play Commentary Prediction

```bash
# fit
python scripts/pbp_classifier.py fit \
--data.labeled_transcript manually-labeled-data \
--trainer.log_every_n_steps 5 \
--trainer.callbacks+=EarlyStopping \
--trainer.callbacks.monitor val_loss \
--trainer.callbacks+=ModelCheckpoint \
--trainer.callbacks.monitor val_loss

# test
python scripts/pbp_classifier.py test \
--config path/to/fit/config.yaml \
--ckpt_path path/to/fit/checkpoints/checkpoint.ckpt

# predict one
python scripts/pbp_classifier.py predict \
--config path/to/fit/config.yaml \
--ckpt_path path/to/fit/checkpoints/checkpoint.ckpt \
--data.predict_transcript_file file-to-predict.json \
--trainer.callbacks+=PBPPredWriter \
--trainer.callbacks.output_file prediction.json

# bulk predict
python scripts/soccernet_annotate_pbp.py \
--pbp_classifier_config_path path/to/fit/config.yaml \
--pbp_classifier_ckpt_path path/to/fit/checkpoints/epoch=9-step=1020.ckpt \
--transcription_dir path/to/SoccerNet-transcribed \
--output_dir path/to/SoccerNet-pbp
```

### Generate TGLG Annotations

```bash
python scripts/gen_anns_soccernet.py \
--pbp_annotated_dir path/to/SoccerNet-pbp \
--output_file_prefix soccernet-tglg \
--output_dir path/to/SoccerNet/
```

## Video Frame Pre-encoding

### Generate Video Statistics Files

We need to know the total number of frames and average frame rate for each video in order to interleave text and frame tokens properly. Unfortunately, this is an expensive operation, so it's best to pre-calculate.

```bash
python scripts/calc_video_stats.py \
--real_time_dataset real_time_vlm_benchmark.datasets.HoloAssistDataset \
--real_time_dataset.video_dir_path /path/to/HoloAssist/HoloAssist/video_pitch_shifted/ \
--real_time_dataset.ann_file_path /path/to/HoloAssist/holo-assist-tglg-eval.json \
--out_file /path/to/HoloAssist/video_stats.json
```

```bash
# set --dataset to the desired dataset
torchrun --nnodes={num_nodes} --nproc_per_node={num_gpus} scripts/videollm_online_encode_frames.py \
--dataset real_time_vlm_benchmark.datasets.holo_assist.HoloAssistDataset \
--dataset.video_dir_path /path/to/HoloAssist/video_pitch_shifted/ \
--dataset.ann_file_path /path/to/HoloAssist/data-annotation-trainval-v1_1.json \
--video_stats_file /path/to/HoloAssist/video_stats.json \
--results_dir path/to/encoded_frames/HoloAssist
```

## Fine-tune on SoccerNet

### VideoLLM-Online

Takes about 2 hours on 4xL40S.

```bash
WANDB_PROJECT=videollm-online-soccernet-train torchrun --nnodes=1 --nproc_per_node=4 scripts/train_videollm_online_soccernet.py \
--live_version live1+ \
--soccernet_dir path/to/SoccerNet \
--video_frame_dir path/to/SoccerNet/encoded-frames \
--video_stats_file path/to/SoccerNet/video_stats.json \
--pretrained_videollm_online chenjoya/videollm-online-8b-v1plus \
--bf16 true \
--report_to wandb \
--num_train_epochs 5 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--gradient_checkpointing True \
--learning_rate 0.0001 \
--optim adamw_torch \
--lr_scheduler_type cosine \
--warmup_ratio 0.05 \
--logging_steps 10 \
--dataloader_num_workers 8 \
--output_dir outputs/videollm-online-soccernet
```

### VLM-TSI

Takes about 2 1/2 hours on 4xL40S.

```bash
WANDB_PROJECT=real-time-soccernet-train torchrun --nnodes=1 --nproc_per_node=4 --tee 3 --log-dir real-time-soccernet-training-logs scripts/train_videollm_online_soccernet.py \
--live_version live1+ \
--videollm_online_variant real-time \
--soccernet_dir path/to/SoccerNet \
--video_frame_dir path/to/SoccerNet/encoded-frames \
--video_stats_file path/to/SoccerNet/video_stats.json \
--pretrained_videollm_online chenjoya/videollm-online-8b-v1plus \
--bf16 true \
--report_to wandb \
--save_strategy epoch \
--eval_strategy epoch \
--num_train_epochs 5 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--gradient_checkpointing True \
--per_device_eval_batch_size 2 \
--learning_rate 0.0001 \
--optim adamw_torch \
--lr_scheduler_type cosine \
--warmup_ratio 0.05 \
--logging_steps 10 \
--dataloader_num_workers 8 \
--output_dir outputs/real-time-soccernet
```

## Fine-tune on Ego4D GoalStep

### VLM-TSI

Takes about a day on 4xL40S.

```bash
WANDB_PROJECT=real-time-ego4d-goalstep-train torchrun --nnodes=1 --nproc_per_node=4 --tee 3 --log-dir real-time-ego4d-goalstep-training-logs scripts/train_videollm_online_ego4d_goalstep.py \
--live_version live1+ \
--videollm_online_variant real-time \
--video_dir path/to/ego4d/v2/videos \
--ann_file path/to/videollm-online-chat-ego4d-134k/goalstep_livechat_trainval_filtered_21k.json \
--video_frame_dir path/to/Ego4DGoalStep/encoded-frames \
--video_stats_file path/to/ego4d/video_stats.json \
--pretrained_videollm_online chenjoya/videollm-online-8b-v1plus \
--bf16 true \
--report_to wandb \
--save_strategy epoch \
--num_train_epochs 2 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing True \
--learning_rate 0.0001 \
--optim adamw_torch \
--lr_scheduler_type cosine \
--warmup_ratio 0.05 \
--logging_steps 10 \
--dataloader_num_workers 12 \
--dataloader_prefetch_factor 4 \
--output_dir outputs/ego4d-goalstep+real-time-model
```

Or on Slurm

```bash
python slurm-scripts/submit_train_videollm_online_ego4d_goalstep.py \
--job_name ego4d-goalstep+real-time-model \
--account <account> \
--partition <partition> \
--time 01-00:00:00 \
--num_gpus 4 \
--mem_per_gpu 32G \
--wandb_project real-time-ego4d-goalstep-train \
--num_dataloader_workers 4 \
--email <email> \
--hf_home /path/to/hf/home \
--train_args '{"live_version": "live1+", "videollm_online_variant": "real-time", "video_dir": "path/to/ego4d/v2/videos", "ann_file": "path/to/videollm-online-chat-ego4d-134k/goalstep_livechat_trainval_filtered_21k.json", "video_frame_dir": "path/to/Ego4DGoalStep/encoded-frames", "video_stats_file": "path/to/ego4d/video_stats.json", "pretrained_videollm_online": "chenjoya/videollm-online-8b-v1plus", "bf16": "true", "report_to": "wandb", "save_strategy": "epoch", "num_train_epochs": "2", "per_device_train_batch_size": "8", "gradient_accumulation_steps": "2", "gradient_checkpointing": "True", "learning_rate": "0.0001", "optim": "adamw_torch", "lr_scheduler_type": "cosine", "warmup_ratio": "0.05", "logging_steps": "10", "output_dir": "outputs/ego4d-goalstep+real-time-model"}'
```

## Run Inference

Below is an example for VLM-TSI (RealTimeSoccerNet) on SoccerNet, but the script is designed to be able to handle all the models and datasets. The script is also designed to be run on a single GPU, which you can do by running the script with plain `python` instead of `torchrun`.

```bash
torchrun --nnodes=1 --nproc_per_node=4 --tee 3 --log-dir path/to/log/dir scripts/run_inference.py \
--model real_time_vlm_benchmark.baseline_models.videollm_online_models.RealTimeSoccerNetModel \
--model.checkpoint path/to/real-time-soccernet/checkpoint \
--model.video_stats_file path/to/SoccerNet/video_stats.json \
--dataset real_time_vlm_benchmark.datasets.SoccerNetDataset \
--dataset.video_dir_path path/to/SoccerNet/ \
--dataset.ann_file_path path/to/SoccerNet/soccernet-tglg-test.json \
--dataset.video_frame_dir_path path/to/SoccerNet/encoded-frames \
--gen_config.max_new_tokens 128 \
--wandb_project real-time-model-soccernet-inference \
--results_dir path/to/inference-results/soccernet+real-time-model \
--wandb_run_name soccernet+real-time-model
```

To run inference with VideoLLM-Online, simply pass in the following options:

```bash
--model real_time_vlm_benchmark.baseline_models.videollm_online_models.VideoLLMOnlineHoloAssistModel \
--model.video_stats_file path/to/SoccerNet/video_stats.json \
```

To run inference on HoloAssist, simply pass in the following options:

```bash
--dataset real_time_vlm_benchmark.datasets.HoloAssistDataset \
--dataset.ann_file_path path/to/HoloAssist/holo-assist-tglg-eval.json \
--dataset.video_dir_path path/to/HoloAssist/video_pitch_shifted/ \
```

## Run Evaluation

You can run evaluations for inference results logged locally or on WandB and report the results locally or to WandB in a flexible manner.

For example, to report SoccerNet evaluation results locally and to WandB based on inference results from the local filesystem or WandB:

```bash
python scripts/eval.py \
--ground_truth_file path/to/SoccerNet/soccernet-tglg-test.json \
--infer_local.files '[soccernet+vlm-tsi.csv]' \
--infer_wandb.entity your-wandb-entity \
--infer_wandb.project your-wandb-project \
--infer_wandb.run_name_regex "videollm-online" \
--eval_wandb.project your-wandb-eval-project \
--eval_local.files_dir path/to/eval_results
```

For HoloLens, simply specify the path to `holo-assist-tglg-eval.json`.
