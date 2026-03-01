#!/bin/bash
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -q gpu
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -J eval

/data/home/xmju/miniconda3/envs/agent/bin/python offline_eval.py