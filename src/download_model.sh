#!/bin/bash
# Fetch microsoft/deberta-xlarge-mnli for notebook 03.
#
# Uses curl rather than huggingface_hub: recent hub versions download through Xet
# storage, which on a slow or flaky connection stalls and then restarts from zero
# instead of resuming. `curl -C -` resumes for real, and the progress is visible.
#
#   ./download_model.sh [target_dir]
#   HF_TOKEN=hf_xxx ./download_model.sh      # optional, lifts the anonymous rate limit
#
# Point notebook 03 at the target directory:  bf.TopicScorer(..., model_type=<dir>)
set -u
DIR="${1:-$HOME/models/deberta-xlarge-mnli}"
REPO="microsoft/deberta-xlarge-mnli"
FILES="config.json tokenizer_config.json vocab.json merges.txt bpe_encoder.bin pytorch_model.bin"
AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")

mkdir -p "$DIR"
for f in $FILES; do
    echo "--- $f"
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        curl -L --fail --retry 5 --retry-delay 3 --retry-all-errors \
             --speed-time 30 --speed-limit 1000 \
             -C - "${AUTH[@]}" -o "$DIR/$f" \
             "https://huggingface.co/$REPO/resolve/main/$f" && break
        echo "  stalled or failed, resuming (attempt $attempt)"
        sleep 3
    done
done
echo
ls -la "$DIR"
