#!/bin/bash

python3 whisper_main.py --dataset_name "dipco" \
                   --model_id "openai/whisper-large-v3" \
                   --version "vanilla" \
                   --environment "bwcluster" \
                   --train_state "T" \
                   --device "cuda" \
                   --task "transcribe" \
                   --developer_mode "N" \
                   --augmentation "N" \
                   --additional_tokens "N"
                   
                   

