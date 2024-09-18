#!/bin/bash

python3 Whisper.py --dataset_name "Chime6" \
                   --model_id "openai/whisper-large-v3" \
                   --version "vanilla" \
                   --environment "bwcluster" \
                   --train_state "T" \
                   --device "cuda" \
                   --task "transcribe" \
                   --developer_mode "N" \
                   --augmentation "Y"
