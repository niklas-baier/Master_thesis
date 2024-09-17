#!/bin/bash

python3 Whisper.py --dataset_name "dipco" \
                   --model_id "openai/whisper-tiny" \
                   --version "peft" \
                   --environment "laptop" \
                   --train_state "T" \
                   --device "cuda" \
                   --task "transcribe" \
                   --developer_mode "Y" \
                   --augmentation "Y"
