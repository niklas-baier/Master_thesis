#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 ../whisper_main.py --dataset_name "dipco" \
                   --model_id "openai/whisper-large-v3" \
                   --version "peft" \
                   --environment "bwcluster" \
                   --train_state "NT" \
                   --device "cuda" \
                   --task "transcribe" \
                   --developer_mode "N" \
                   --augmentation "N" \
                   --additional_tokens "N"\
                   --run_notes "baseline for whisper large"\
		   --dataset_evaluation_part "eval"\
		   --oversampling_clean_data 1\
                   --data_portion "all"\
		   --beamforming "N"\
		   --SWAD False\
		   --diffusion "N" \
		   --checkpoint ""



