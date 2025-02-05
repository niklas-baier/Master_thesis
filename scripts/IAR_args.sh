#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python3 ../whisper_main.py --dataset_name "dipco" \
                   --model_id "distil-whisper/distil-large-v3" \
                   --version "vanilla" \
                   --environment "cluster" \
                   --train_state "NT" \
                   --device "cuda" \
                   --task "transcribe" \
                   --developer_mode "N" \
                   --augmentation "N" \
                   --additional_tokens "N"\
                   --run_notes "first try with diffusion_model"\
		   --dataset_evaluation_part "eval"\
		   --oversampling_clean_data 1\
                   --data_portion "far-only"\
		   --beamforming "Y"\
		   --SWAD False



