#!/bin/bash
#SBATCH --job-name="miranda9job"
#SBATCH --output="demo.%j.%N.out"
#SBATCH --error="demo.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=8
#SBATCH --threads-per-core=4
#SBATCH --mem-per-cpu=1200
#SBATCH --export=ALL
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-user=brando.science@gmail.com
#SBATCH --mail-type=ALL

srun hostname

import torch
import torchvision

import higher
import torchmeta

import time

print('HelloWorld')

secs = 30
time.sleep(secs)