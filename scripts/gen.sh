#!/bin/bash

#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -R 'order[r15s]'
#BSUB -R 'select[hname!=gpu13&&hname!=gpu14&&hname!=gpu15&&hname!=gpu16&&hname!=gpu17&&hname!=gpu18]'
#BSUB -J debug
#BSUB -oo output_dir/gen/debug/out.txt
#BSUB -eo output_dir/gen/debug/err.txt

mkdir -p output_dir/gen/debug
rm -rf output_dir/gen/debug/*

python3 gen_img_uncond.py \
--model mage_vit_base_patch16 \
--temp 0.0 \
--num_iter 100 \
--ckpt output_dir/train/260128-170553_overfit_gen/checkpoint-last.pth \
--batch_size 1 \
--num_images 1 \
--output_dir output_dir/gen/debug
