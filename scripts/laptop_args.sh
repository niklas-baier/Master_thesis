#!/bin/bash

python3 ../whisper_main.py --dataset_name "dipco" \
                   --model_id "openai/whisper-tiny" \
                   --version "vanilla" \
                   --environment "laptop" \
                   --train_state "NT" \
                   --device "cuda" \
                   --task "transcribe" \
                   --developer_mode "Y" \
                   --augmentation "N" \
                   --additional_tokens "N"\
                   --run_notes "correction wer"
