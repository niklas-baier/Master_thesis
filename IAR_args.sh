#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python3 Whisper.py --dataset_name "dipco" \
                   --model_id "distil-whisper/distil-large-v3" \
                   --version "peft" \
                   --environment "laptop" \
                   --train_state "T" \
                   --device "cuda" \
                   --task "transcribe" \
                   --developer_mode "N" \
                   --augmentation "Y"
