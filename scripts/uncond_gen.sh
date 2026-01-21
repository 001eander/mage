#!/bin/bash

#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -R 'order[r15s]'
#BSUB -R 'select[hname!=gpu13&&hname!=gpu14&&hname!=gpu15&&hname!=gpu16&&hname!=gpu17&&hname!=gpu18]'
#BSUB -J uncond_gen
#BSUB -oo output_dir/uncond_gen/out.txt
#BSUB -eo output_dir/uncond_gen/err.txt

uv run gen_img_uncond.py --temp 6.0 --num_iter 20 \
--ckpt pre_train_ckpt/mage-vitb-1600.pth --batch_size 32 --num_images 5 \
--model mage_vit_base_patch16 --output_dir output_dir/uncond_gen

