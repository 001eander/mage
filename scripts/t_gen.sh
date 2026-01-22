#!/bin/bash

#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -R 'order[r15s]'
#BSUB -R 'select[hname!=gpu13&&hname!=gpu14&&hname!=gpu15&&hname!=gpu16&&hname!=gpu17&&hname!=gpu18]'
#BSUB -J NAME
#BSUB -oo output_dir/gen/NAME/out.txt
#BSUB -eo output_dir/gen/NAME/err.txt

mkdir -p output_dir/gen/NAME

uv run gen_img_uncond.py \
--image_size 32 \
--model mage_vit_base_patch16 \
--temp 0.0 \
--num_iter 100 \
--ckpt output_dir/train/NAME/checkpoint-last.pth \
--batch_size 2 \
--num_images 2 \
--output_dir output_dir/gen/NAME
