#!/bin/bash

python3 edges_training.py --dataset_name "edges" \
                   --model_size "smallu" \
		   --batch_size 1 \
		   --num_epochs 50 \
		   --ema 0.99 \
		   --weight_decay 0.001 \
		   --lr 0.00001 \
		   --scheduler_type "one_cycle" \
                   --environment "bwcluster" \
		   --checkpoint "" \
		   --run_notes "this is the first try "\
		   --subset_ratio "0.1"



