#!/bin/bash

#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -R 'order[ut]'
#BSUB -J demo
#BSUB -oo output_dir/demo/out.txt
#BSUB -eo output_dir/demo/err.txt

.venv/bin/python -V
.venv/bin/python demo.py
nvidia-smi
