#!/bin/bash

#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -R 'order[r15s]'
#BSUB -R 'select[hname!=gpu13&&hname!=gpu14&&hname!=gpu15&&hname!=gpu16&&hname!=gpu17&&hname!=gpu18]'
#BSUB -J one
#BSUB -oo output_dir/train/one/out.txt
#BSUB -eo output_dir/train/one/err.txt

mkdir -p output_dir/train/one
rm -rf output_dir/train/one/*

python3 main_pretrain.py \
--mask_ratio_mu 0.8 \
--mask_ratio_std 0 \
--disable_aug \
--subset_n 1 \
--model mage_vit_base_patch16 \
--data_path ~/data/tiny-imagenet-200 \
--output_dir ./output_dir/train/one \
--epochs 100 \
--batch_size 1 \