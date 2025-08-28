#!/bin/bash

python3 edges_training.py --dataset_name "edges" \
                   --model_size "smallu" \
		   --batch_size 4 \
		   --num_epochs 100 \
		   --ema 0.99 \
		   --weight_decay 0.001 \
		   --lr 0.00001 \
		   --scheduler_type "one_cycle" \
                   --environment "bwcluster" \
		   --checkpoint "" \
		   --run_notes "checkpoitns "\
		   --subset_ratio "0.1"



