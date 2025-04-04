#!/usr/bin/env bash

INPUT_DIR="$1"
OUTPUT_DIR="$2"
DRY_RUN="${3:-false}"

reencode() {
  local INPUT_FILE="$1"
  local REL_PATH
  REL_PATH=$(realpath --relative-to="$INPUT_DIR" "$INPUT_FILE")
  local OUTPUT_FILE="$OUTPUT_DIR/$REL_PATH"
  local LOG_DIR="./reencode_logs"
  mkdir -p "$LOG_DIR"

  local SKIP_LOG="$LOG_DIR/skipped.log"
  local REENCODE_LOG="$LOG_DIR/reencoded.log"
  local CORRUPT_LOG="$LOG_DIR/corrupted.log"

  # Check if output exists
  if [[ -f "$OUTPUT_FILE" ]]; then
    # Check if output is readable
    if ! ffprobe -v error -i "$OUTPUT_FILE" -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 > /dev/null 2>&1; then
      echo "Corrupted: $REL_PATH — would re-encode" | tee -a "$CORRUPT_LOG"
      $DRY_RUN && return
    else
      local IN_DUR OUT_DUR DIFF
      IN_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT_FILE")
      OUT_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_FILE")

      if [[ -n "$IN_DUR" && -n "$OUT_DUR" ]]; then
        DIFF=$(awk -v dur_in="$IN_DUR" -v dur_out="$OUT_DUR" 'BEGIN { print (dur_in - dur_out > 0 ? dur_in - dur_out : dur_out - dur_in) }')
        if awk -v d="$DIFF" 'BEGIN { exit !(d <= 0.5) }'; then
          echo "Already encoded: $REL_PATH" | tee -a "$SKIP_LOG"
          return
        else
          echo "Duration mismatch: $REL_PATH — would re-encode" | tee -a "$REENCODE_LOG"
          $DRY_RUN && return
        fi
      else
        echo "Unreadable duration: $REL_PATH — would re-encode" | tee -a "$REENCODE_LOG"
        $DRY_RUN && return
      fi
    fi
  else
    echo "New file: $REL_PATH — would encode" | tee -a "$REENCODE_LOG"
    $DRY_RUN && return
  fi

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
export DRY_RUN

find "$INPUT_DIR" -type f -iname "*.mp4" | parallel --bar -j"$(nproc)" reencode {}
