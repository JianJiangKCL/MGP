#!/bin/bash



cd ..

SAVE_PATH='results/new_checkpoints2'


GPU_ID=1
DATASET_PATH=''


CONFIG='configs/second_perceiver.yaml'

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -c ${CONFIG}  --results_dir $SAVE_PATH -t wandb_mode="online"

