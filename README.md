# real-time-vlm-benchmark

## Installation

```bash
# If Slurm, ensure you have the latest CUDA and GCC loaded
module load cuda gcc

# Create a virtual env, e.g., using uv
uv venv

# flash-attn build dependencies
uv pip install torch setuptools psutil packaging ninja

# Build flash-attn
uv pip install flash-attn --no-build-isolation

# Install python dependencies
# Specify --extra flash-attn if necessary
uv sync

# Workaround for https://github.com/OpenNMT/CTranslate2/issues/1826
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib/
```

## HoloAssist Real-Time Annotation Generation

```bash
python scripts/gen_anns_holo_assist.py \
--holo_assist_video_dir /path/to/HoloAssist/video_pitch_shifted/ \
--holo_assist_anns_file /path/to/HoloAssist/data-annotation-trainval-v1_1.json \
--output_file /path/to/HoloAssist/real-time-eval-annotation.json
```

## SoccerNet Real-Time Annotation Generation

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

### Generate Real-Time Annotations

```bash
python scripts/gen_anns_soccernet.py \
--pbp_annotated_dir path/to/SoccerNet-pbp \
--output_file path/to/SoccerNet/real-time-eval-annotation.json
```
