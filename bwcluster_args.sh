#!/bin/bash

python3 Whisper.py --dataset_name "dipco" \
                   --model_id "openai/whisper-large-v3" \
                   --version "peft" \
                   --environment "bwcluster" \
                   --train_state "T" \
                   --device "cuda" \
                   --task "transcribe" \
                   --developer_mode "N" \
                   --augmentation "Y"
