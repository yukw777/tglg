#!/usr/bin/env bash

INPUT_DIR="$1"
OUTPUT_DIR="$2"

reencode() {
  local INPUT_FILE="$1"
  local REL_PATH
  REL_PATH=$(realpath --relative-to="$INPUT_DIR" "$INPUT_FILE")
  local OUTPUT_FILE="$OUTPUT_DIR/$REL_PATH"
  mkdir -p "$(dirname "$OUTPUT_FILE")"
  ffmpeg -y -loglevel error -i "$INPUT_FILE" \
    -vf scale=384:384 \
    -c:v libx264 -preset veryfast -crf 28 \
    -c:a aac -b:a 128k \
    "$OUTPUT_FILE"
}

export -f reencode
export INPUT_DIR
export OUTPUT_DIR

find "$INPUT_DIR" -type f -iname "*.mp4" | parallel --bar -j"$(nproc)" reencode {}
