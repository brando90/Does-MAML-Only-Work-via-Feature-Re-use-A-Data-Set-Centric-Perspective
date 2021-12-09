#!/bin/bash
# pkill -9 python
# pkill -U miranda9

# -- setup up for condor_submit background script in vision-cluster
export HOME=/home/miranda9
# to have modules work and the conda command work
source /etc/bashrc
source /etc/profile
source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/.bash_profile

module load gcc/9.2.0
#module load cuda-toolkit/10.2
module load cuda-toolkit/11.1

# some quick checks
#conda activate synthesis
conda activate metalearning_gpu
#export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=1
nvcc --version
hostname
which python
python -c "import uutils; print(uutils); uutils.hello()"
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo ---- Running your python main ----

# set experiment id
export SLURM_JOBID=$(((RANDOM)))
echo SLURM_JOBID=$SLURM_JOBID
export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
echo OUT_FILE=$OUT_FILE

#pip install wandb --upgrade

# - run experimen
python -u ~/automl-meta-learning/automl-proj-src/experiments/meta_learning/main_metalearning.py
#python -u ~/automl-meta-learning/results_plots_sl_vs_ml/fall2021/main_dist_sl_vs_maml.py
echo pid = $!

echo "Done with bash script (experiment or dispatched daemon experiments). "
