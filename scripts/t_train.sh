#!/bin/bash

#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -R 'order[r15s]'
#BSUB -R 'select[hname!=gpu13&&hname!=gpu14&&hname!=gpu15&&hname!=gpu16&&hname!=gpu17&&hname!=gpu18]'
#BSUB -J NAME
#BSUB -oo output_dir/train/NAME/out.txt
#BSUB -eo output_dir/train/NAME/err.txt

mkdir -p output_dir/train/NAME

uv run main_pretrain.py \
--input_size 32 \
--model mage_vit_base_patch16 \
--data_path ~/data/flower \
--output_dir ./output_dir/train/NAME \
--epochs 500 \
--batch_size 2 \